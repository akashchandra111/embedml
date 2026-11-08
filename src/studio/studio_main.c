/*
 * elml_studio: a CUI desktop frontend for the embedlml trainer.
 *
 * The window is split in two. The left panel edits the same parameters the
 * command-line trainer accepts; the right panel shows live loss and accuracy
 * charts while a background worker runs the training session.
 *
 * Only the UI thread calls CUI. The worker owns the training session and
 * publishes an epoch snapshot plus the chart history under a mutex. Stopping
 * is an atomic flag checked at batch boundaries inside elml_training_run().
 */
#include <cui/cui.h>
#include <cui/cui_app.h>
#include <cui/cui_icons.h>

#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <threads.h>

#include "elml/model.h"
#include "elml/registry.h"
#include "elml/types.h"
#include "train/config.h"
#include "train/csv.h"
#include "train/session.h"

#ifndef ELML_STUDIO_EXAMPLE_CONFIG
#define ELML_STUDIO_EXAMPLE_CONFIG ""
#endif

#define STUDIO_PATH_MAX ELML_CONFIG_PATH_MAX
#define STUDIO_LIST_MAX ELML_CONFIG_LIST_MAX
#define STUDIO_HISTORY_MAX 8192
#define STUDIO_ARG_MAX 48

typedef struct studio_snapshot {
  bool prepared;
  bool finished;
  bool exported;
  bool early_stopped;
  bool stopped;
  bool has_accuracy;
  bool has_val;
  uint32_t epoch;
  uint32_t epochs;
  uint32_t best_epoch;
  uint32_t train_rows;
  uint32_t val_rows;
  uint32_t input_dim;
  uint32_t output_dim;
  float loss;
  float accuracy;
  float best_loss;
  float final_train_loss;
  float final_train_accuracy;
  float final_val_loss;
  float final_val_accuracy;
  bool has_final_accuracy;
  char model[256];
  char algorithm[96];
  char export_path[STUDIO_PATH_MAX];
  char status[1536];
  char error[1536];
} studio_snapshot;

typedef struct studio_view {
  studio_snapshot snap;
  bool running;
  int history_count;
  bool has_accuracy_history;
} studio_view;

typedef struct studio_columns {
  char **names;   /* header names, or NULL entries for headerless files */
  char **labels;  /* "name (index)" shown by the multi-select */
  bool *input_selected;
  bool *output_selected;
  bool *input_previous;
  bool *output_previous;
  int count;
  bool loaded;
  bool checked;
  bool has_header;
  bool truncated;
  int unresolved_inputs;
  int unresolved_outputs;
  char path[STUDIO_PATH_MAX];
  char delimiter[8];
  char status[512];
} studio_columns;

typedef struct studio_state {
  cui_app *app;
  mtx_t mutex;
  thrd_t worker;
  bool worker_started;

  bool running;
  bool finished;
  elml_training *active;
  elml_config pending;
  studio_snapshot snap;

  studio_columns columns;

  float loss_history[STUDIO_HISTORY_MAX];
  float accuracy_history[STUDIO_HISTORY_MAX];
  int history_count;
  bool has_accuracy_history;

  /* Parameters shown in the sidebar. */
  char config_path[STUDIO_PATH_MAX];
  char csv_path[STUDIO_PATH_MAX];
  char val_csv_path[STUDIO_PATH_MAX];
  char export_path[STUDIO_PATH_MAX];
  char inputs[STUDIO_LIST_MAX];
  char outputs[STUDIO_LIST_MAX];
  char layers[128];
  char delimiter[8];
  bool has_header;
  int hidden_activation;
  int loss_index;
  int optimizer_index;
  int missing_index;
  double split;
  double seed;
  double epochs;
  double batch;
  double learning_rate;
  double l2;
  double early_stop;

  cui_theme_mode theme;
} studio_state;

/* Copied out of the mutex once per frame so the charts never point into the
 * worker's half-written history. */
static float studio_draw_loss[STUDIO_HISTORY_MAX];
static float studio_draw_accuracy[STUDIO_HISTORY_MAX];

static const char *const studio_activation_labels[] = {
    "ReLU", "Leaky ReLU", "Tanh", "Sigmoid", "Linear"};
static const char *const studio_activation_values[] = {
    "relu", "leaky_relu", "tanh", "sigmoid", "linear"};
static const char *const studio_loss_labels[] = {
    "MSE (regression)", "MAE (regression)",
    "Cross entropy (classification)"};
static const char *const studio_loss_values[] = {"mse", "mae", "cce"};
static const char *const studio_optimizer_labels[] = {"Adam", "SGD"};
static const char *const studio_optimizer_values[] = {"adam", "sgd"};
static const char *const studio_missing_labels[] = {"Error", "Zero", "Column mean"};
static const char *const studio_missing_values[] = {"error", "zero", "mean"};

#define STUDIO_COUNT(a) ((int)(sizeof(a) / sizeof((a)[0])))

static void studio_lock(studio_state *state) { (void)mtx_lock(&state->mutex); }
static void studio_unlock(studio_state *state) { (void)mtx_unlock(&state->mutex); }

static bool studio_is_running(studio_state *state)
{
  studio_lock(state);
  const bool running = state->running;
  studio_unlock(state);
  return running;
}

static void studio_set_error(studio_state *state, const char *format, ...)
{
  va_list args;
  studio_lock(state);
  va_start(args, format);
  vsnprintf(state->snap.error, sizeof state->snap.error, format, args);
  va_end(args);
  studio_unlock(state);
}

static void studio_clear_run_locked(studio_state *state)
{
  memset(&state->snap, 0, sizeof state->snap);
  state->history_count = 0;
  state->has_accuracy_history = false;
  state->finished = false;
}

static void studio_defaults(studio_state *state)
{
  state->theme = CUI_THEME_DARK;
  state->has_header = true;
  state->hidden_activation = 0;
  state->loss_index = 0;
  state->optimizer_index = 0;
  state->missing_index = 0;
  state->split = 0.8;
  state->seed = 42.0;
  state->epochs = 100.0;
  state->batch = 16.0;
  state->learning_rate = 0.001;
  state->l2 = 0.0;
  state->early_stop = 20.0;
  snprintf(state->delimiter, sizeof state->delimiter, "%s", ",");
  snprintf(state->layers, sizeof state->layers, "%s", "16");
  snprintf(state->export_path, sizeof state->export_path, "%s", "model.elml");
  snprintf(state->config_path, sizeof state->config_path, "%s",
           ELML_STUDIO_EXAMPLE_CONFIG);
  snprintf(state->snap.status, sizeof state->snap.status, "%s",
           "Set data paths, then start training.");
}

static void studio_dirname(const char *path, char *out, size_t cap)
{
  const char *slash = strrchr(path, '/');
#ifdef _WIN32
  const char *backslash = strrchr(path, '\\');
  if (backslash != nullptr && (slash == nullptr || backslash > slash)) {
    slash = backslash;
  }
#endif
  if (slash == nullptr) {
    snprintf(out, cap, "%s", ".");
  } else if (slash == path) {
    snprintf(out, cap, "%s", "/");
  } else {
    const size_t length = (size_t)(slash - path);
    if (length >= cap) {
      snprintf(out, cap, "%s", ".");
      return;
    }
    memcpy(out, path, length);
    out[length] = '\0';
  }
}

static bool studio_path_is_absolute(const char *path)
{
  if (path[0] == '/') {
    return true;
  }
#ifdef _WIN32
  if (isalpha((unsigned char)path[0]) && path[1] == ':') {
    return true;
  }
#endif
  return false;
}

static void studio_copy_string(char *out, size_t cap, const char *text)
{
  if (cap == 0u) {
    return;
  }
  if (text == nullptr) {
    out[0] = '\0';
    return;
  }
  size_t length = strlen(text);
  if (length >= cap) {
    length = cap - 1u;
  }
  memcpy(out, text, length);
  out[length] = '\0';
}

static void studio_append_string(char *out, size_t cap, const char *text)
{
  const size_t used = strlen(out);
  if (used + 1u >= cap) {
    return;
  }
  size_t length = strlen(text);
  const size_t room = cap - used - 1u;
  if (length > room) {
    length = room;
  }
  memcpy(out + used, text, length);
  out[used + length] = '\0';
}

static void studio_resolve_path(const char *base, const char *path, char *out,
                                size_t cap)
{
  if (path == nullptr || path[0] == '\0') {
    out[0] = '\0';
    return;
  }
  if (studio_path_is_absolute(path)) {
    studio_copy_string(out, cap, path);
  } else {
    studio_copy_string(out, cap, base);
    studio_append_string(out, cap, "/");
    studio_append_string(out, cap, path);
  }
}

/* ------------------------------------------------------------------------- */
/* CSV column pickers                                                        */
/* ------------------------------------------------------------------------- */

static void studio_columns_release(studio_columns *columns)
{
  if (columns->names != nullptr) {
    for (int i = 0; i < columns->count; ++i) {
      free(columns->names[i]);
    }
  }
  if (columns->labels != nullptr) {
    for (int i = 0; i < columns->count; ++i) {
      free(columns->labels[i]);
    }
  }
  free(columns->names);
  free(columns->labels);
  free(columns->input_selected);
  free(columns->output_selected);
  free(columns->input_previous);
  free(columns->output_previous);
  memset(columns, 0, sizeof *columns);
}

static char studio_delimiter_value(const char *text)
{
  if (text == nullptr || text[0] == '\0') {
    return '\0';
  }
  if (text[1] == '\0') {
    return text[0];
  }
  if (strcmp(text, "\\t") == 0) {
    return '\t';
  }
  return '\0';
}

static bool studio_columns_signature_matches(const studio_state *state)
{
  return state->columns.checked &&
         strcmp(state->columns.path, state->csv_path) == 0 &&
         strcmp(state->columns.delimiter, state->delimiter) == 0 &&
         state->columns.has_header == state->has_header;
}

static void studio_columns_note_signature(studio_state *state)
{
  studio_columns *columns = &state->columns;
  studio_copy_string(columns->path, sizeof columns->path, state->csv_path);
  studio_copy_string(columns->delimiter, sizeof columns->delimiter,
                     state->delimiter);
  columns->has_header = state->has_header;
  columns->checked = true;
}

/* Build the comma-separated 1-based index list the trainer expects. */
static bool studio_columns_join(const studio_columns *columns,
                                const bool *selected, char *out, size_t cap,
                                int *count_out)
{
  size_t used = 0u;
  int count = 0;
  out[0] = '\0';

  for (int i = 0; i < columns->count; ++i) {
    if (!selected[i]) {
      continue;
    }
    char number[16];
    const int written = snprintf(number, sizeof number, "%s%d",
                                 (count == 0) ? "" : ",", i + 1);
    if (written < 0) {
      return false;
    }
    const size_t length = (size_t)written;
    if (used + length + 1u > cap) {
      out[0] = '\0';
      return false;
    }
    memcpy(out + used, number, length);
    used += length;
    out[used] = '\0';
    count++;
  }

  if (count_out != nullptr) {
    *count_out = count;
  }
  return true;
}

static void studio_columns_sync_lists(studio_state *state)
{
  studio_columns *columns = &state->columns;
  if (!columns->loaded) {
    return;
  }

  int input_count = 0;
  int output_count = 0;
  const bool inputs_ok =
      studio_columns_join(columns, columns->input_selected, state->inputs,
                          sizeof state->inputs, &input_count);
  const bool outputs_ok =
      studio_columns_join(columns, columns->output_selected, state->outputs,
                          sizeof state->outputs, &output_count);
  columns->truncated = !inputs_ok || !outputs_ok;

  int written = snprintf(columns->status, sizeof columns->status,
                         "%s: %d columns. %d input%s, %d output%s selected.",
                         columns->has_header ? "Header" : "No header",
                         columns->count, input_count,
                         (input_count == 1) ? "" : "s", output_count,
                         (output_count == 1) ? "" : "s");
  if (written < 0) {
    columns->status[0] = '\0';
    written = 0;
  }
  const int unresolved =
      columns->unresolved_inputs + columns->unresolved_outputs;
  if (unresolved > 0 && (size_t)written < sizeof columns->status) {
    written += snprintf(columns->status + written,
                        sizeof columns->status - (size_t)written,
                        " %d configured column%s not found.", unresolved,
                        (unresolved == 1) ? " was" : "s were");
  }
  if (columns->truncated && (size_t)written < sizeof columns->status) {
    (void)snprintf(columns->status + written,
                   sizeof columns->status - (size_t)written,
                   " Too many columns selected to fit the configuration "
                   "list.");
  }
}

static int studio_columns_lookup(const studio_columns *columns,
                                 const char *token)
{
  if (columns->has_header && columns->names != nullptr) {
    for (int i = 0; i < columns->count; ++i) {
      if (columns->names[i] != nullptr &&
          strcmp(columns->names[i], token) == 0) {
        return i;
      }
    }
  }

  errno = 0;
  char *end = nullptr;
  const unsigned long value = strtoul(token, &end, 10);
  if (errno == 0 && end != token && *end == '\0' && value >= 1ul &&
      value <= (unsigned long)columns->count) {
    return (int)(value - 1ul);
  }
  return -1;
}

static void studio_columns_apply_list(const studio_columns *columns,
                                      const char *list, bool *selected,
                                      int *unresolved)
{
  memset(selected, 0, (size_t)columns->count * sizeof(bool));

  const char *cursor = list;
  while (*cursor != '\0') {
    while (*cursor == ' ' || *cursor == '\t') {
      ++cursor;
    }
    if (*cursor == '\0') {
      break;
    }

    const char *end = cursor;
    while (*end != '\0' && *end != ',') {
      ++end;
    }
    const char *trimmed = end;
    while (trimmed > cursor && (trimmed[-1] == ' ' || trimmed[-1] == '\t')) {
      --trimmed;
    }
    const size_t length = (size_t)(trimmed - cursor);
    if (length > 0u && length < 128u) {
      char token[128];
      memcpy(token, cursor, length);
      token[length] = '\0';
      const int index = studio_columns_lookup(columns, token);
      if (index >= 0) {
        selected[index] = true;
      } else {
        (*unresolved)++;
      }
    } else if (length >= 128u) {
      (*unresolved)++;
    }

    cursor = (*end == ',') ? end + 1 : end;
  }
}

static bool studio_columns_load(studio_state *state, bool force)
{
  studio_columns *columns = &state->columns;
  if (!force && columns->loaded && studio_columns_signature_matches(state)) {
    return true;
  }

  if (state->csv_path[0] == '\0') {
    studio_columns_release(columns);
    snprintf(columns->status, sizeof columns->status,
             "Choose a training CSV to list its columns.");
    return false;
  }

  const char delimiter = studio_delimiter_value(state->delimiter);
  if (delimiter == '\0') {
    studio_columns_release(columns);
    snprintf(columns->status, sizeof columns->status,
             "Delimiter must be one character, or \\t for a tab.");
    studio_columns_note_signature(state);
    return false;
  }

  elml_csv header;
  elml_status status = elml_csv_read_header(state->csv_path, delimiter,
                                            &header);
  if (status != ELML_OK) {
    studio_columns_release(columns);
    snprintf(columns->status, sizeof columns->status,
             "Could not read columns from %.300s (%.100s).", state->csv_path,
             elml_error_string(status));
    studio_columns_note_signature(state);
    return false;
  }
  if (header.cols > (uint32_t)INT_MAX) {
    elml_csv_free(&header);
    studio_columns_release(columns);
    snprintf(columns->status, sizeof columns->status,
             "The CSV header has too many columns for the picker.");
    studio_columns_note_signature(state);
    return false;
  }

  const int count = (int)header.cols;
  studio_columns_release(columns);
  columns->names = (char **)calloc((size_t)count, sizeof(char *));
  columns->labels = (char **)calloc((size_t)count, sizeof(char *));
  columns->input_selected = (bool *)calloc((size_t)count, sizeof(bool));
  columns->output_selected = (bool *)calloc((size_t)count, sizeof(bool));
  columns->input_previous = (bool *)calloc((size_t)count, sizeof(bool));
  columns->output_previous = (bool *)calloc((size_t)count, sizeof(bool));
  columns->count = count;

  bool allocated = columns->names != nullptr && columns->labels != nullptr &&
                   columns->input_selected != nullptr &&
                   columns->output_selected != nullptr &&
                   columns->input_previous != nullptr &&
                   columns->output_previous != nullptr;
  if (allocated) {
    for (int i = 0; i < count; ++i) {
      const char *cell = elml_csv_cell(&header, 0u, (uint32_t)i);
      if (state->has_header && cell != nullptr && cell[0] != '\0') {
        const size_t length = strlen(cell);
        columns->names[i] = (char *)malloc(length + 1u);
        if (columns->names[i] == nullptr) {
          allocated = false;
          break;
        }
        memcpy(columns->names[i], cell, length + 1u);
      }

      char fallback[32];
      const char *label = columns->names[i];
      if (label == nullptr) {
        snprintf(fallback, sizeof fallback, "Column %d", i + 1);
        label = fallback;
      }
      char *copy = (char *)malloc(strlen(label) + 32u);
      if (copy == nullptr) {
        allocated = false;
        break;
      }
      snprintf(copy, strlen(label) + 32u, "%s (%d)", label, i + 1);
      columns->labels[i] = copy;
    }
  }

  if (!allocated) {
    elml_csv_free(&header);
    studio_columns_release(columns);
    snprintf(columns->status, sizeof columns->status,
             "Out of memory while reading the CSV header.");
    studio_columns_note_signature(state);
    return false;
  }

  elml_csv_free(&header);
  columns->loaded = true;
  studio_columns_note_signature(state);

  columns->unresolved_inputs = 0;
  columns->unresolved_outputs = 0;
  studio_columns_apply_list(columns, state->inputs, columns->input_selected,
                            &columns->unresolved_inputs);
  studio_columns_apply_list(columns, state->outputs, columns->output_selected,
                            &columns->unresolved_outputs);
  /* Inputs win if a configuration names the same column twice. */
  for (int i = 0; i < columns->count; ++i) {
    if (columns->input_selected[i] && columns->output_selected[i]) {
      columns->output_selected[i] = false;
    }
  }
  memcpy(columns->input_previous, columns->input_selected,
         (size_t)columns->count * sizeof(bool));
  memcpy(columns->output_previous, columns->output_selected,
         (size_t)columns->count * sizeof(bool));
  studio_columns_sync_lists(state);
  return true;
}

static void studio_columns_sync(studio_state *state)
{
  if (state->csv_path[0] == '\0' && !state->columns.loaded) {
    return;
  }
  if (studio_columns_signature_matches(state)) {
    return;
  }
  (void)studio_columns_load(state, false);
}

static bool studio_columns_validate(studio_state *state, char *error,
                                    size_t cap)
{
  if (!studio_columns_load(state, false)) {
    studio_copy_string(error, cap,
                       state->columns.status[0] != '\0'
                           ? state->columns.status
                           : "Could not read the CSV header.");
    return false;
  }

  int input_count = 0;
  int output_count = 0;
  for (int i = 0; i < state->columns.count; ++i) {
    if (state->columns.input_selected[i]) {
      input_count++;
    }
    if (state->columns.output_selected[i]) {
      output_count++;
    }
  }
  if (input_count == 0) {
    snprintf(error, cap, "Select at least one input column.");
    return false;
  }
  if (output_count == 0) {
    snprintf(error, cap, "Select at least one output column.");
    return false;
  }
  const int unresolved =
      state->columns.unresolved_inputs + state->columns.unresolved_outputs;
  if (unresolved > 0) {
    snprintf(error, cap,
             "%d configured column%s not found in the CSV header. Choose "
             "them again with the picker.",
             unresolved, (unresolved == 1) ? " was" : "s were");
    return false;
  }
  for (int i = 0; i < state->columns.count; ++i) {
    if (state->columns.input_selected[i] &&
        state->columns.output_selected[i]) {
      snprintf(error, cap,
               "Column %d is selected as both an input and an output. "
               "Choose each column for one side only.",
               i + 1);
      return false;
    }
  }

  /* The trainer resolves header names before numeric indices. If a selected
   * column's 1-based number is also a header name elsewhere, the emitted index
   * would name the wrong column. Refuse instead of training on bad data. */
  if (state->columns.has_header) {
    for (int i = 0; i < state->columns.count; ++i) {
      if (!state->columns.input_selected[i] &&
          !state->columns.output_selected[i]) {
        continue;
      }
      char token[16];
      snprintf(token, sizeof token, "%d", i + 1);
      for (int j = 0; j < state->columns.count; ++j) {
        if (j == i || state->columns.names[j] == nullptr) {
          continue;
        }
        if (strcmp(state->columns.names[j], token) == 0) {
          snprintf(error, cap,
                   "Column %d is selected, but another header is literally "
                   "named \"%.100s\". Column numbers would be ambiguous; "
                   "rename that header.",
                   i + 1, state->columns.names[j]);
          return false;
        }
      }
    }
  }

  if (!studio_columns_join(&state->columns, state->columns.input_selected,
                           state->inputs, sizeof state->inputs, nullptr) ||
      !studio_columns_join(&state->columns, state->columns.output_selected,
                           state->outputs, sizeof state->outputs, nullptr)) {
    snprintf(error, cap,
             "Too many columns are selected to fit the configuration list. "
             "Choose fewer inputs or outputs.");
    return false;
  }
  return true;
}

static int studio_activation_index(elml_activation activation)
{
  for (int i = 0; i < STUDIO_COUNT(studio_activation_values); ++i) {
    if (strcmp(studio_activation_values[i],
               elml_activation_name(activation)) == 0) {
      return i;
    }
  }
  return 0;
}

static void studio_apply_config(studio_state *state, const elml_config *config)
{
  char base[STUDIO_PATH_MAX];
  studio_dirname(state->config_path, base, sizeof base);

  studio_resolve_path(base, config->csv_path, state->csv_path,
                      sizeof state->csv_path);
  studio_resolve_path(base, config->val_csv_path, state->val_csv_path,
                      sizeof state->val_csv_path);
  studio_resolve_path(base, config->export_path, state->export_path,
                      sizeof state->export_path);

  snprintf(state->inputs, sizeof state->inputs, "%s", config->inputs);
  snprintf(state->outputs, sizeof state->outputs, "%s", config->outputs);
  state->has_header = config->has_header;
  if (config->delimiter == '\t') {
    snprintf(state->delimiter, sizeof state->delimiter, "%s", "\\t");
  } else {
    state->delimiter[0] = config->delimiter;
    state->delimiter[1] = '\0';
  }
  state->missing_index = (int)config->missing;
  state->split = config->split;
  state->seed = (double)config->seed;
  state->loss_index = (int)config->loss;
  state->optimizer_index = (int)config->optimizer;
  state->epochs = (double)config->epochs;
  state->batch = (double)config->batch;
  state->learning_rate = config->learning_rate;
  state->l2 = config->l2;
  state->early_stop = (double)config->early_stop;

  size_t used = 0u;
  state->layers[0] = '\0';
  for (uint32_t i = 0u; i < config->hidden_count; ++i) {
    const int written = snprintf(state->layers + used,
                                 sizeof state->layers - used, "%s%u",
                                 (i == 0u) ? "" : ",", config->hidden[i]);
    if (written < 0 || (size_t)written >= sizeof state->layers - used) {
      break;
    }
    used += (size_t)written;
  }

  int activation = 0;
  if (config->hidden_count > 0u && config->activation_count > 0u) {
    activation = studio_activation_index(config->activations[0]);
  }
  state->hidden_activation = activation;
}

static void studio_load_config(studio_state *state, bool quiet)
{
  if (studio_is_running(state)) {
    return;
  }
  if (state->config_path[0] == '\0') {
    if (!quiet) {
      studio_set_error(state, "Choose a .ini configuration file first.");
    }
    return;
  }

  char *argv[3];
  argv[0] = "elml_studio";
  argv[1] = "--config";
  argv[2] = state->config_path;

  elml_config config;
  const elml_status status = elml_config_load(&config, 3, argv);
  if (status != ELML_OK) {
    if (!quiet) {
      studio_set_error(state,
                       "Could not load %s. Check the file and the terminal "
                       "output for the exact reason.",
                       state->config_path);
    }
    return;
  }

  studio_apply_config(state, &config);
  (void)studio_columns_load(state, true);
  studio_lock(state);
  studio_clear_run_locked(state);
  snprintf(state->snap.status, sizeof state->snap.status, "Loaded %s",
           state->config_path);
  studio_unlock(state);
}

static bool studio_layers_valid(const char *text, uint32_t *count_out)
{
  const char *cursor = text;
  uint32_t count = 0u;

  while (*cursor != '\0') {
    while (*cursor == ' ' || *cursor == '\t') {
      ++cursor;
    }
    if (*cursor == '\0') {
      break;
    }
    if (*cursor == ',') {
      return false;
    }

    errno = 0;
    char *end = nullptr;
    const unsigned long value = strtoul(cursor, &end, 10);
    if (end == cursor || errno == ERANGE || value == 0u ||
        value > 1000000ul) {
      return false;
    }
    ++count;

    cursor = end;
    while (*cursor == ' ' || *cursor == '\t') {
      ++cursor;
    }
    if (*cursor == ',') {
      ++cursor;
      continue;
    }
    if (*cursor != '\0') {
      return false;
    }
  }

  if (count == 0u || count > ELML_MAX_HIDDEN_LAYERS) {
    return false;
  }
  *count_out = count;
  return true;
}

static bool studio_build_config(studio_state *state, elml_config *config,
                                char *error, size_t cap)
{
  if (state->csv_path[0] == '\0') {
    snprintf(error, cap, "Choose a training CSV file.");
    return false;
  }
  if (state->export_path[0] == '\0') {
    snprintf(error, cap, "Choose where to export the model.");
    return false;
  }
  if (!studio_columns_validate(state, error, cap)) {
    return false;
  }

  uint32_t layer_count = 0u;
  if (!studio_layers_valid(state->layers, &layer_count)) {
    snprintf(error, cap,
             "Hidden layers must be 1..%u positive widths separated by "
             "commas, for example 128,64.",
             (unsigned)ELML_MAX_HIDDEN_LAYERS);
    return false;
  }
  if (state->split < 0.05 || state->split > 0.99) {
    snprintf(error, cap, "Validation split must be between 0.05 and 0.99.");
    return false;
  }
  if (state->epochs < 1.0 || state->batch < 1.0) {
    snprintf(error, cap, "Epochs and batch size must be at least 1.");
    return false;
  }
  if (state->learning_rate <= 0.0) {
    snprintf(error, cap, "Learning rate must be positive.");
    return false;
  }
  if (state->l2 < 0.0) {
    snprintf(error, cap, "L2 regularization cannot be negative.");
    return false;
  }
  if (state->delimiter[0] == '\0') {
    snprintf(error, cap, "Delimiter cannot be empty (use , or \\t).");
    return false;
  }

  const int activation =
      (state->hidden_activation >= 0 &&
       state->hidden_activation < STUDIO_COUNT(studio_activation_values))
          ? state->hidden_activation
          : 0;
  const int loss =
      (state->loss_index >= 0 && state->loss_index < STUDIO_COUNT(studio_loss_values))
          ? state->loss_index
          : 0;
  const int optimizer =
      (state->optimizer_index >= 0 &&
       state->optimizer_index < STUDIO_COUNT(studio_optimizer_values))
          ? state->optimizer_index
          : 0;
  const int missing =
      (state->missing_index >= 0 &&
       state->missing_index < STUDIO_COUNT(studio_missing_values))
          ? state->missing_index
          : 0;

  char activations[STUDIO_LIST_MAX];
  size_t used = 0u;
  activations[0] = '\0';
  for (uint32_t i = 0u; i < layer_count; ++i) {
    const int written =
        snprintf(activations + used, sizeof activations - used, "%s%s",
                 (i == 0u) ? "" : ",", studio_activation_values[activation]);
    if (written < 0 || (size_t)written >= sizeof activations - used) {
      snprintf(error, cap, "Activation list is too long.");
      return false;
    }
    used += (size_t)written;
  }

  char split_text[32];
  char seed_text[32];
  char epochs_text[32];
  char batch_text[32];
  char learning_rate_text[64];
  char l2_text[64];
  char early_stop_text[32];
  char missing_text[16];
  char loss_text[16];
  char optimizer_text[16];
  studio_copy_string(missing_text, sizeof missing_text,
                     studio_missing_values[missing]);
  studio_copy_string(loss_text, sizeof loss_text, studio_loss_values[loss]);
  studio_copy_string(optimizer_text, sizeof optimizer_text,
                     studio_optimizer_values[optimizer]);
  snprintf(split_text, sizeof split_text, "%.12g", state->split);
  snprintf(seed_text, sizeof seed_text, "%llu",
           (unsigned long long)(state->seed < 0.0 ? 0.0 : state->seed));
  snprintf(epochs_text, sizeof epochs_text, "%u",
           (unsigned)(state->epochs < 1.0 ? 1.0 : state->epochs));
  snprintf(batch_text, sizeof batch_text, "%u",
           (unsigned)(state->batch < 1.0 ? 1.0 : state->batch));
  snprintf(learning_rate_text, sizeof learning_rate_text, "%.12g",
           state->learning_rate);
  snprintf(l2_text, sizeof l2_text, "%.12g", state->l2);
  snprintf(early_stop_text, sizeof early_stop_text, "%u",
           (unsigned)(state->early_stop < 0.0 ? 0.0 : state->early_stop));

  char *argv[STUDIO_ARG_MAX];
  int argc = 0;
  argv[argc++] = "elml_studio";
  argv[argc++] = "--csv";
  argv[argc++] = state->csv_path;
  if (state->val_csv_path[0] != '\0') {
    argv[argc++] = "--val-csv";
    argv[argc++] = state->val_csv_path;
  }
  argv[argc++] = "--inputs";
  argv[argc++] = state->inputs;
  argv[argc++] = "--outputs";
  argv[argc++] = state->outputs;
  argv[argc++] = (state->has_header) ? "--header" : "--no-header";
  argv[argc++] = "--delimiter";
  argv[argc++] = state->delimiter;
  argv[argc++] = "--missing";
  argv[argc++] = missing_text;
  argv[argc++] = "--val";
  argv[argc++] = split_text;
  argv[argc++] = "--seed";
  argv[argc++] = seed_text;
  argv[argc++] = "--layers";
  argv[argc++] = state->layers;
  argv[argc++] = "--activations";
  argv[argc++] = activations;
  argv[argc++] = "--loss";
  argv[argc++] = loss_text;
  argv[argc++] = "--optimizer";
  argv[argc++] = optimizer_text;
  argv[argc++] = "--epochs";
  argv[argc++] = epochs_text;
  argv[argc++] = "--batch";
  argv[argc++] = batch_text;
  argv[argc++] = "--lr";
  argv[argc++] = learning_rate_text;
  argv[argc++] = "--l2";
  argv[argc++] = l2_text;
  argv[argc++] = "--early-stop";
  argv[argc++] = early_stop_text;
  argv[argc++] = "--export";
  argv[argc++] = state->export_path;

  const elml_status status = elml_config_load(config, argc, argv);
  if (status != ELML_OK) {
    snprintf(error, cap,
             "The configuration was rejected. Check the layers, column "
             "names and numeric values (the terminal shows the exact "
             "reason).");
    return false;
  }
  return true;
}

static void studio_history_append_locked(studio_state *state,
                                         const elml_train_metrics *metrics)
{
  if (state->history_count >= STUDIO_HISTORY_MAX) {
    const int half = STUDIO_HISTORY_MAX / 2;
    for (int i = 0; i < half; ++i) {
      state->loss_history[i] = state->loss_history[i * 2];
      state->accuracy_history[i] = state->accuracy_history[i * 2];
    }
    state->history_count = half;
  }
  const int index = state->history_count;
  state->loss_history[index] = metrics->loss;
  state->accuracy_history[index] = metrics->accuracy;
  state->history_count = index + 1;
  state->has_accuracy_history = metrics->has_accuracy;
}

static void studio_progress(const elml_train_metrics *metrics, void *user)
{
  studio_state *state = (studio_state *)user;
  studio_lock(state);
  state->snap.epoch = metrics->epoch;
  state->snap.epochs = metrics->epochs;
  state->snap.best_epoch = metrics->best_epoch;
  state->snap.loss = metrics->loss;
  state->snap.accuracy = metrics->accuracy;
  state->snap.has_accuracy = metrics->has_accuracy;
  state->snap.train_rows = metrics->train_rows;
  state->snap.val_rows = metrics->val_rows;
  state->snap.early_stopped = metrics->early_stopped;
  state->snap.stopped = metrics->stopped;
  studio_history_append_locked(state, metrics);
  studio_unlock(state);
}

static int studio_worker(void *user)
{
  studio_state *state = (studio_state *)user;
  elml_training training;
  elml_status status = elml_training_init(&training, &state->pending);

  if (status != ELML_OK) {
    studio_lock(state);
    state->active = nullptr;
    state->running = false;
    state->finished = true;
    snprintf(state->snap.error, sizeof state->snap.error, "%s",
             (training.error[0] != '\0') ? training.error
                                         : elml_error_string(status));
    snprintf(state->snap.status, sizeof state->snap.status, "%s",
             "Could not prepare the training data.");
    studio_unlock(state);
    elml_training_destroy(&training);
    return 0;
  }

  studio_lock(state);
  state->active = &training;
  state->snap.prepared = true;
  state->snap.train_rows = training.train.rows;
  state->snap.val_rows = training.val.rows;
  state->snap.input_dim = training.train.input_dim;
  state->snap.output_dim = training.train.output_dim;
  state->snap.has_val = training.val.rows > 0u;
  state->snap.has_accuracy = training.train.class_index != nullptr;
  state->snap.best_loss = training.best_loss;
  snprintf(state->snap.export_path, sizeof state->snap.export_path, "%s",
           training.config.export_path);

  size_t used = 0u;
  const int first = snprintf(state->snap.model, sizeof state->snap.model,
                             "%u", training.train.input_dim);
  if (first > 0) {
    used = (size_t)first;
  }
  for (uint32_t i = 0u; i < training.net.layer_count; ++i) {
    const int written = snprintf(state->snap.model + used,
                                 sizeof state->snap.model - used, " -> %u",
                                 training.net.layers[i].out_dim);
    if (written < 0 || (size_t)written >= sizeof state->snap.model - used) {
      break;
    }
    used += (size_t)written;
  }
  snprintf(state->snap.algorithm, sizeof state->snap.algorithm, "%s + %s",
           elml_loss_name(training.config.loss),
           (training.config.optimizer == ELML_OPTIMIZER_ADAM) ? "Adam" : "SGD");
  snprintf(state->snap.status, sizeof state->snap.status,
           "Training %u rows (%u validation). Stop any time to keep the best "
           "epoch.",
           training.train.rows, training.val.rows);
  studio_unlock(state);

  status = elml_training_run(&training, studio_progress, state);

  bool exported = false;
  char error[512];
  error[0] = '\0';
  if (status == ELML_OK) {
    if (training.ran) {
      const elml_status export_status = elml_training_export(&training);
      if (export_status == ELML_OK) {
        exported = true;
      } else {
        snprintf(error, sizeof error, "%s",
                 (training.error[0] != '\0') ? training.error
                                             : elml_error_string(export_status));
      }
    } else {
      snprintf(error, sizeof error, "%s",
               "Stopped before the first epoch finished; no model was "
               "exported.");
    }
  } else {
    snprintf(error, sizeof error, "%s",
             (training.error[0] != '\0') ? training.error
                                         : elml_error_string(status));
  }

  studio_lock(state);
  state->active = nullptr;
  state->running = false;
  state->finished = true;
  state->snap.finished = true;
  state->snap.exported = exported;
  state->snap.early_stopped = training.early_stopped;
  state->snap.stopped = training.stopped;
  state->snap.best_epoch = training.best_epoch;
  state->snap.best_loss = training.best_loss;
  state->snap.epoch = training.epochs_done;
  state->snap.epochs = training.config.epochs;
  state->snap.has_accuracy = training.train.class_index != nullptr;
  state->snap.has_final_accuracy = training.has_final_accuracy;
  state->snap.final_train_loss = training.final_train_loss;
  state->snap.final_train_accuracy = training.final_train_accuracy;
  state->snap.final_val_loss = training.final_val_loss;
  state->snap.final_val_accuracy = training.final_val_accuracy;
  if (training.ran) {
    /* The exported model is the best-weight snapshot, so the KPI cards should
     * describe that model rather than the last epoch that ran. */
    if (training.val.rows > 0u) {
      state->snap.loss = training.final_val_loss;
      state->snap.accuracy = training.final_val_accuracy;
    } else {
      state->snap.loss = training.final_train_loss;
      state->snap.accuracy = training.final_train_accuracy;
    }
  }
  if (exported) {
    if (training.early_stopped) {
      snprintf(state->snap.status, sizeof state->snap.status,
               "Early stopped at epoch %u (best epoch %u). Exported %s.",
               training.epochs_done, training.best_epoch,
               training.config.export_path);
    } else if (training.stopped) {
      snprintf(state->snap.status, sizeof state->snap.status,
               "Stopped at epoch %u (best epoch %u). Exported %s.",
               training.epochs_done, training.best_epoch,
               training.config.export_path);
    } else {
      snprintf(state->snap.status, sizeof state->snap.status,
               "Training complete after %u epochs (best epoch %u). Exported "
               "%s.",
               training.epochs_done, training.best_epoch,
               training.config.export_path);
    }
  } else if (status == ELML_OK && !training.ran) {
    snprintf(state->snap.status, sizeof state->snap.status, "%s",
             "Stopped before the first epoch; nothing was exported.");
  } else if (status == ELML_OK) {
    snprintf(state->snap.status, sizeof state->snap.status,
             "Training finished, but the model could not be exported.");
  } else {
    snprintf(state->snap.status, sizeof state->snap.status, "%s",
             "Training failed.");
  }
  snprintf(state->snap.error, sizeof state->snap.error, "%s", error);
  studio_unlock(state);

  elml_training_destroy(&training);
  return 0;
}

static void studio_start(studio_state *state)
{
  if (studio_is_running(state)) {
    return;
  }
  if (state->worker_started) {
    (void)thrd_join(state->worker, nullptr);
    state->worker_started = false;
  }

  elml_config config;
  char error[256];
  if (!studio_build_config(state, &config, error, sizeof error)) {
    studio_set_error(state, "%s", error);
    return;
  }

  state->pending = config;
  studio_lock(state);
  studio_clear_run_locked(state);
  state->running = true;
  state->active = nullptr;
  snprintf(state->snap.status, sizeof state->snap.status, "%s",
           "Preparing data and network...");
  snprintf(state->snap.export_path, sizeof state->snap.export_path, "%s",
           config.export_path);
  studio_unlock(state);

  if (thrd_create(&state->worker, studio_worker, state) != thrd_success) {
    studio_lock(state);
    state->running = false;
    snprintf(state->snap.error, sizeof state->snap.error, "%s",
             "Could not create the training thread.");
    snprintf(state->snap.status, sizeof state->snap.status, "%s",
             "Training did not start.");
    studio_unlock(state);
    return;
  }
  state->worker_started = true;
}

static void studio_stop(studio_state *state)
{
  studio_lock(state);
  if (state->active != nullptr) {
    elml_training_request_stop(state->active);
    snprintf(state->snap.status, sizeof state->snap.status, "%s",
             "Stopping at the next batch boundary...");
  }
  studio_unlock(state);
}

static void studio_shutdown(studio_state *state)
{
  studio_lock(state);
  if (state->active != nullptr) {
    elml_training_request_stop(state->active);
  }
  studio_unlock(state);
  if (state->worker_started) {
    (void)thrd_join(state->worker, nullptr);
    state->worker_started = false;
  }
}

static void studio_take_view(studio_state *state, studio_view *view)
{
  studio_lock(state);
  view->snap = state->snap;
  view->running = state->running;
  view->history_count = state->history_count;
  view->has_accuracy_history = state->has_accuracy_history;
  if (view->history_count > 0) {
    memcpy(studio_draw_loss, state->loss_history,
           (size_t)view->history_count * sizeof(float));
    memcpy(studio_draw_accuracy, state->accuracy_history,
           (size_t)view->history_count * sizeof(float));
  }
  studio_unlock(state);

  if (view->history_count == 1) {
    studio_draw_loss[1] = studio_draw_loss[0];
    studio_draw_accuracy[1] = studio_draw_accuracy[0];
    view->history_count = 2;
  }
}

/* ------------------------------------------------------------------------- */
/* Sidebar                                                                   */
/* ------------------------------------------------------------------------- */

static void studio_section(cui_ctx *ui, const char *title)
{
  cui_gap(ui, 6);
  cui_text(ui, title, .font = CUI_FONT_MEDIUM, .size = 13,
           .color = CUI_COLOR_TEXT_SECONDARY);
}

static void studio_text_field(cui_ctx *ui, const char *id, const char *label,
                              char *buffer, size_t cap, const char *placeholder,
                              const char *hint, const char *tooltip,
                              bool disabled)
{
  (void)cui_input(ui, id, buffer, cap, .label = label,
                  .placeholder = placeholder, .hint = hint,
                  .width = CUI_FILL, .disabled = disabled);
  if (tooltip != nullptr) {
    cui_tooltip(ui, tooltip);
  }
}

static void studio_path_field(cui_ctx *ui, studio_state *state, const char *id,
                              const char *label, char *buffer, size_t cap,
                              cui_file_dialog_kind kind,
                              const char *filter_name, const char *filter,
                              const char *tooltip, bool disabled)
{
  cui_row(ui, .width = CUI_FILL, .gap = 8, .align = CUI_ALIGN_END) {
    (void)cui_input(ui, id, buffer, cap, .label = label, .width = CUI_FILL,
                    .disabled = disabled);
    if (tooltip != nullptr) {
      cui_tooltip(ui, tooltip);
    }
    if (cui_icon_button(ui, CUI_ICON_FOLDER_OPEN, .disabled = disabled,
                        .tooltip = "Browse")) {
      char picked[STUDIO_PATH_MAX];
      picked[0] = '\0';
      if (cui_app_file_dialog(
              state->app,
              &(cui_file_dialog_opts){.kind = kind,
                                      .title = label,
                                      .filter_name = filter_name,
                                      .filter = filter},
              picked, sizeof picked)) {
        snprintf(buffer, cap, "%s", picked);
      }
    }
  }
}

static void studio_number_field(cui_ctx *ui, const char *id, const char *label,
                                double *value, double min, double max,
                                double step, int decimals, const char *suffix,
                                const char *hint, const char *tooltip,
                                bool disabled)
{
  (void)cui_number(ui, id, value, min, max, .label = label, .step = step,
                   .decimals = decimals, .suffix = suffix,
                   .disabled = disabled, .width = CUI_FILL);
  if (tooltip != nullptr) {
    cui_tooltip(ui, tooltip);
  }
  if (hint != nullptr) {
    cui_muted(ui, hint);
  }
}

static void studio_column_picker(cui_ctx *ui, studio_state *state,
                                 bool running)
{
  studio_columns *columns = &state->columns;

  if (!columns->loaded) {
    studio_text_field(ui, "inputs", "Input columns", state->inputs,
                      sizeof state->inputs, "sepal_length,sepal_width,...",
                      "Column names are matched against the CSV header. The "
                      "picker appears once the file can be read.",
                      "Comma-separated feature columns; names match the CSV "
                      "header, or 1-based numbers for headerless files.",
                      running);
    studio_text_field(ui, "outputs", "Output column(s)", state->outputs,
                      sizeof state->outputs, "species",
                      "One name for classification, or a comma-separated "
                      "list for regression.",
                      "Target column(s); names match the CSV header, or "
                      "1-based numbers for headerless files.",
                      running);
    if (cui_button(ui, "Read CSV columns", .icon = CUI_ICON_REFRESH_CW,
                   .variant = CUI_BUTTON_GHOST, .size = CUI_SIZE_SM,
                   .tooltip = "Read the first CSV row and populate the column "
                              "pickers",
                   .disabled = running)) {
      (void)studio_columns_load(state, true);
    }
    if (columns->status[0] != '\0') {
      cui_muted(ui, columns->status);
    }
    return;
  }

  const bool searchable = columns->count > 8;
  bool inputs_changed = false;
  bool outputs_changed = false;

  cui_disabled_scope(ui, running) {
    inputs_changed = cui_multi_select_ex(
        ui, "column_inputs", (const char *const *)columns->labels,
        columns->count, columns->input_selected,
        &(cui_multi_select_opts){
            .label = "Input columns",
            .placeholder = "Choose one or more inputs",
            .searchable = searchable,
            .max_chips = 2,
            .width = CUI_FILL});
    cui_tooltip(ui,
                "Feature columns for the model. Choosing one here removes it "
                "from the outputs.");

    outputs_changed = cui_multi_select_ex(
        ui, "column_outputs", (const char *const *)columns->labels,
        columns->count, columns->output_selected,
        &(cui_multi_select_opts){
            .label = "Output column(s)",
            .placeholder = "Choose the target columns",
            .searchable = searchable,
            .max_chips = 2,
            .width = CUI_FILL});
    cui_tooltip(ui,
                "Target columns for the model. Choosing one here removes it "
                "from the inputs.");
  }

  if (inputs_changed) {
    for (int i = 0; i < columns->count; ++i) {
      if (columns->input_selected[i] && !columns->input_previous[i]) {
        columns->output_selected[i] = false;
      }
    }
  }
  if (outputs_changed) {
    for (int i = 0; i < columns->count; ++i) {
      if (columns->output_selected[i] && !columns->output_previous[i]) {
        columns->input_selected[i] = false;
      }
    }
  }
  if (inputs_changed || outputs_changed) {
    if (inputs_changed) {
      columns->unresolved_inputs = 0;
    }
    if (outputs_changed) {
      columns->unresolved_outputs = 0;
    }
    studio_columns_sync_lists(state);
  }
  memcpy(columns->input_previous, columns->input_selected,
         (size_t)columns->count * sizeof(bool));
  memcpy(columns->output_previous, columns->output_selected,
         (size_t)columns->count * sizeof(bool));

  if (cui_button(ui, "Rescan CSV columns", .icon = CUI_ICON_REFRESH_CW,
                 .variant = CUI_BUTTON_GHOST, .size = CUI_SIZE_SM,
                 .tooltip = "Re-read the CSV header after editing the file",
                 .disabled = running)) {
    (void)studio_columns_load(state, true);
  }
  if (columns->status[0] != '\0') {
    cui_muted(ui, columns->status);
  }
}

static bool studio_sidebar(cui_ctx *ui, studio_state *state, bool running)
{
  bool load_clicked = false;
  const float width = (cui_viewport_size(ui).x < 980.0f) ? 340.0f : 390.0f;

  cui_column(ui, .width = CUI_PX(width), .height = CUI_FILL,
             .bg = CUI_COLOR_SURFACE) {
    cui_column(ui, .width = CUI_FILL, .pad = 20, .gap = 4) {
      cui_row(ui, .width = CUI_FILL, .align = CUI_ALIGN_CENTER, .gap = 10) {
        cui_icon(ui, CUI_ICON_LAYERS, 22, CUI_COLOR_ACCENT);
        cui_text(ui, "embedlml", .font = CUI_FONT_DISPLAY, .size = 20);
      }
      cui_secondary(ui, "Training studio");
    }
    cui_separator(ui);

    cui_scroll(ui, .id = "studio_params", .width = CUI_FILL,
               .height = CUI_FILL, .pad = 20, .gap = 14, .fade_edges = true) {
      studio_section(ui, "CONFIGURATION");
      cui_row(ui, .width = CUI_FILL, .gap = 8, .align = CUI_ALIGN_END) {
        (void)cui_input(ui, "config_path", state->config_path,
                        sizeof state->config_path,
                        .label = "Config file (.ini)", .width = CUI_FILL,
                        .disabled = running);
        cui_tooltip(ui, "Path to a training configuration. Load fills every "
                        "field below from this file.");
        if (cui_icon_button(ui, CUI_ICON_FOLDER_OPEN, .disabled = running,
                            .tooltip = "Browse for an .ini file")) {
          char picked[STUDIO_PATH_MAX];
          picked[0] = '\0';
          if (cui_app_file_dialog(
                  state->app,
                  &(cui_file_dialog_opts){
                      .kind = CUI_FILE_OPEN,
                      .title = "Open training configuration",
                      .filter_name = "INI files",
                      .filter = "ini"},
                  picked, sizeof picked)) {
            snprintf(state->config_path, sizeof state->config_path, "%s",
                     picked);
          }
        }
        if (cui_button(ui, "Load", .icon = CUI_ICON_REFRESH_CW,
                       .tooltip = "Load the .ini and fill every field without "
                                  "starting training",
                       .disabled = running)) {
          load_clicked = true;
        }
      }
      cui_muted(ui, "Loading an .ini fills every field below. Relative paths "
                    "in the file are resolved next to it.");

      studio_section(ui, "DATA");
      studio_path_field(ui, state, "csv_path", "Training CSV",
                        state->csv_path, sizeof state->csv_path,
                        CUI_FILE_OPEN, "CSV files", "csv",
                        "CSV file with the training data. Its header fills the "
                        "column pickers below.",
                        running);
      studio_path_field(ui, state, "val_csv_path", "Validation CSV (optional)",
                        state->val_csv_path, sizeof state->val_csv_path,
                        CUI_FILE_OPEN, "CSV files", "csv",
                        "Optional separate validation file. Leave empty to "
                        "split the training CSV using Split.",
                        running);
      studio_column_picker(ui, state, running);
      (void)cui_switch(ui, "First row is a header", &state->has_header,
                       .disabled = running);
      cui_tooltip(ui, "Treat the first CSV row as column names instead of "
                      "data. Rescan the pickers after changing this.");
      cui_row(ui, .width = CUI_FILL, .gap = 12) {
        studio_number_field(ui, "split", "Split", &state->split,
                            0.05, 0.99, 0.05, 2, nullptr, nullptr,
                            "Fraction of rows used for training. 0.80 means "
                            "80% train and 20% validation.",
                            running);
        studio_number_field(ui, "seed", "Seed", &state->seed, 0.0,
                            4294967295.0, 1.0, 0, nullptr, nullptr,
                            "Random seed for the split and weight "
                            "initialization. Change it for a different run.",
                            running);
      }
      cui_muted(ui, "0 picks a fixed seed too; change it for a new shuffle.");
      cui_row(ui, .width = CUI_FILL, .gap = 12, .align = CUI_ALIGN_END) {
        (void)cui_input(ui, "delimiter", state->delimiter,
                        sizeof state->delimiter, .label = "Delimiter",
                        .width = CUI_FILL, .disabled = running);
        cui_tooltip(ui, "Field separator in the CSV: one character, or \\t "
                        "for a tab.");
        (void)cui_select(ui, "missing", studio_missing_labels,
                         STUDIO_COUNT(studio_missing_labels),
                         &state->missing_index, .label = "Missing values",
                         .width = CUI_FILL, .disabled = running);
        cui_tooltip(ui, "How empty CSV cells are handled while loading.");
      }

      studio_section(ui, "MODEL");
      studio_text_field(ui, "layers", "Hidden layers", state->layers,
                        sizeof state->layers, "128,64",
                        "Comma-separated widths. One hidden layer is fine.",
                        "Comma-separated hidden layer sizes, for example "
                        "128,64. Leave empty for a single linear layer.",
                        running);
      (void)cui_select(ui, "hidden_activation", studio_activation_labels,
                       STUDIO_COUNT(studio_activation_labels),
                       &state->hidden_activation,
                       .label = "Hidden activation", .width = CUI_FILL,
                       .disabled = running);
      cui_tooltip(ui, "Activation applied after each hidden layer.");
      (void)cui_select(ui, "loss", studio_loss_labels,
                       STUDIO_COUNT(studio_loss_labels), &state->loss_index,
                       .label = "Loss", .width = CUI_FILL,
                       .disabled = running);
      cui_tooltip(ui, "Objective the network minimizes. It also chooses the "
                      "output activation.");
      (void)cui_select(ui, "optimizer", studio_optimizer_labels,
                       STUDIO_COUNT(studio_optimizer_labels),
                       &state->optimizer_index, .label = "Optimizer",
                       .width = CUI_FILL, .disabled = running);
      cui_tooltip(ui, "Weight update rule used during training.");
      cui_muted(ui, "The output activation follows the loss: softmax for "
                    "cross entropy, linear otherwise.");

      studio_section(ui, "TRAINING");
      cui_row(ui, .width = CUI_FILL, .gap = 12) {
        studio_number_field(ui, "epochs", "Epochs", &state->epochs, 1.0,
                            1000000.0, 10.0, 0, nullptr, nullptr,
                            "Maximum number of passes over the training set.",
                            running);
        studio_number_field(ui, "batch", "Batch size", &state->batch, 1.0,
                            65536.0, 1.0, 0, nullptr, nullptr,
                            "Rows per mini-batch update. Smaller batches "
                            "update more often.",
                            running);
      }
      cui_row(ui, .width = CUI_FILL, .gap = 12) {
        studio_number_field(ui, "lr", "Learning rate", &state->learning_rate,
                            0.0000001, 100.0, 0.0001, 7, nullptr, nullptr,
                            "Step size for the optimizer. Smaller is slower "
                            "but more stable.",
                            running);
        studio_number_field(ui, "l2", "L2 penalty", &state->l2, 0.0, 100.0,
                            0.0001, 7, nullptr, nullptr,
                            "Weight decay strength. 0 disables it.", running);
      }
      studio_number_field(ui, "early_stop", "Early-stop patience",
                          &state->early_stop, 0.0, 1000000.0, 5.0, 0,
                          "epochs", "0 disables early stopping.",
                          "Stop after this many epochs without validation "
                          "loss improvement. 0 disables early stopping.",
                          running);

      studio_section(ui, "EXPORT");
      studio_path_field(ui, state, "export_path", "Model output",
                        state->export_path, sizeof state->export_path,
                        CUI_FILE_SAVE, "embedlml models", "elml",
                        "Where the best weights are written after training "
                        "finishes or you press Stop.",
                        running);
      cui_muted(ui, "The best weights are exported automatically when "
                    "training finishes or you press Stop.");
    }

    cui_separator(ui);
    cui_column(ui, .width = CUI_FILL, .pad = 16, .gap = 10) {
      if (running) {
        if (cui_button(ui, "Stop and export best", .variant = CUI_BUTTON_DANGER,
                       .icon = CUI_ICON_SQUARE, .full_width = true,
                       .tooltip = "Stop after the current batch, restore the "
                                  "best epoch, and export")) {
          studio_stop(state);
        }
        cui_muted(ui, "The current best epoch is restored and exported.");
      } else {
        if (cui_button(ui, "Start training", .variant = CUI_BUTTON_PRIMARY,
                       .icon = CUI_ICON_PLAY, .full_width = true,
                       .tooltip = "Validate the settings and start training "
                                  "in the background")) {
          studio_start(state);
        }
        cui_muted(ui,
                  "Training runs in the background; the graphs update live.");
      }
    }
  }

  return load_clicked;
}

/* ------------------------------------------------------------------------- */
/* Main panel                                                                */
/* ------------------------------------------------------------------------- */

static void studio_detail(cui_ctx *ui, const char *label, const char *value)
{
  cui_row(ui, .width = CUI_FILL, .gap = 12, .align = CUI_ALIGN_START) {
    cui_text(ui, label, .size = 12, .width = CUI_PX(120),
             .color = CUI_COLOR_TEXT_SECONDARY, .nowrap = true);
    cui_text(ui, value, .size = 13, .width = CUI_FILL);
  }
}

static void studio_metrics(cui_ctx *ui, const studio_view *view)
{
  char epoch_value[48];
  char loss_value[48];
  char accuracy_value[48];
  char best_value[48];

  if (view->snap.epoch > 0u) {
    snprintf(epoch_value, sizeof epoch_value, "%u", view->snap.epoch);
  } else {
    snprintf(epoch_value, sizeof epoch_value, "%s", "-");
  }
  if (view->snap.epoch > 0u) {
    snprintf(loss_value, sizeof loss_value, "%.4f",
             (double)view->snap.loss);
  } else {
    snprintf(loss_value, sizeof loss_value, "%s", "-");
  }
  if (view->snap.epoch > 0u && view->snap.has_accuracy) {
    snprintf(accuracy_value, sizeof accuracy_value, "%.2f%%",
             (double)view->snap.accuracy * 100.0);
  } else {
    snprintf(accuracy_value, sizeof accuracy_value, "%s", "-");
  }
  if (view->snap.best_epoch > 0u) {
    snprintf(best_value, sizeof best_value, "%u", view->snap.best_epoch);
  } else {
    snprintf(best_value, sizeof best_value, "%s", "-");
  }

  const int spark_count = view->history_count;
  const float *loss_spark = (spark_count > 1) ? studio_draw_loss : nullptr;
  const float *accuracy_spark =
      (spark_count > 1 && view->has_accuracy_history) ? studio_draw_accuracy
                                                     : nullptr;

  cui_row(ui, .width = CUI_FILL, .wrap = true, .gap = 12) {
    cui_metric(ui, "Epoch", epoch_value, .icon = CUI_ICON_ACTIVITY,
               .width = CUI_GROW(1));
    cui_metric(ui, "Val loss", loss_value,
               .icon = CUI_ICON_TRENDING_DOWN, .width = CUI_GROW(1),
               .spark = loss_spark, .spark_count = spark_count,
               .lower_is_better = true);
    cui_metric(ui, "Accuracy", accuracy_value, .icon = CUI_ICON_TARGET,
               .width = CUI_GROW(1), .spark = accuracy_spark,
               .spark_count = accuracy_spark != nullptr ? spark_count : 0);
    cui_metric(ui, "Best epoch", best_value, .icon = CUI_ICON_ZAP,
               .width = CUI_GROW(1));
  }
}

static void studio_progress_card(cui_ctx *ui, const studio_view *view)
{
  char label[96];
  float fraction = 0.0f;
  if (view->snap.epochs > 0u) {
    fraction = (float)((double)view->snap.epoch / (double)view->snap.epochs);
  }
  if (view->snap.finished && !view->snap.stopped &&
      !view->snap.early_stopped) {
    fraction = 1.0f;
  }
  if (fraction < 0.0f) {
    fraction = 0.0f;
  }
  if (fraction > 1.0f) {
    fraction = 1.0f;
  }

  if (view->snap.epochs > 0u) {
    snprintf(label, sizeof label, "Epoch %u of %u", view->snap.epoch,
             view->snap.epochs);
  } else if (view->running) {
    snprintf(label, sizeof label, "%s", "Preparing data...");
  } else {
    snprintf(label, sizeof label, "%s", "Ready");
  }

  cui_card(ui, .width = CUI_FILL, .gap = 10) {
    cui_row(ui, .width = CUI_FILL, .align = CUI_ALIGN_CENTER, .gap = 10) {
      cui_text(ui, label, .font = CUI_FONT_MEDIUM);
      cui_spacer(ui);
      if (view->running) {
        cui_spinner(ui, 16.0f);
      }
    }
    cui_progress(ui, fraction,
                 .tone = (view->snap.stopped || view->snap.early_stopped)
                             ? CUI_TONE_WARNING
                             : CUI_TONE_ACCENT,
                 .height = 10.0f);
    if (view->snap.status[0] != '\0') {
      cui_secondary(ui, view->snap.status);
    }
  }
}

static void studio_chart(cui_ctx *ui, const char *title, const char *subtitle,
                         const char *id, const float *values, int count,
                         const char *format, bool available,
                         const char *empty_message)
{
  cui_card(ui, .width = CUI_FILL, .gap = 12) {
    cui_column(ui, .width = CUI_FILL, .gap = 2) {
      cui_text(ui, title, .font = CUI_FONT_MEDIUM, .size = 15);
      cui_text(ui, subtitle, .size = 12, .color = CUI_COLOR_TEXT_SECONDARY);
    }
    if (!available || count <= 0) {
      cui_text(ui, empty_message, .size = 12,
               .color = CUI_COLOR_TEXT_MUTED);
    } else {
      cui_series series[1];
      series[0].name = title;
      series[0].values = values;
      series[0].count = count;
      series[0].color = CUI_COLOR_SERIES(0);
      cui_line_chart(ui, id, series, 1, .height = 210.0f, .area = true,
                     .smooth = true, .format = format, .no_legend = true);
    }
  }
}

static void studio_run_details(cui_ctx *ui, studio_state *state,
                               const studio_view *view)
{
  cui_card(ui, .width = CUI_FILL, .gap = 12) {
    cui_row(ui, .width = CUI_FILL, .align = CUI_ALIGN_CENTER) {
      cui_text(ui, "Run details", .font = CUI_FONT_MEDIUM, .size = 15);
      cui_spacer(ui);
      if (view->snap.exported) {
        cui_badge(ui, "Model exported", .tone = CUI_TONE_SUCCESS,
                  .dot = true);
      }
    }

    if (view->snap.model[0] != '\0') {
      studio_detail(ui, "Model", view->snap.model);
    }
    if (view->snap.algorithm[0] != '\0') {
      studio_detail(ui, "Objective", view->snap.algorithm);
    }
    if (view->snap.train_rows > 0u) {
      char rows[128];
      snprintf(rows, sizeof rows, "%u train / %u validation rows",
               view->snap.train_rows, view->snap.val_rows);
      studio_detail(ui, "Data", rows);
    }
    if (view->snap.finished && view->snap.epoch > 0u) {
      char train_metrics[160];
      char val_metrics[160];
      if (view->snap.has_final_accuracy) {
        snprintf(train_metrics, sizeof train_metrics,
                 "loss %.4f / accuracy %.2f%%",
                 (double)view->snap.final_train_loss,
                 (double)view->snap.final_train_accuracy * 100.0);
      } else {
        snprintf(train_metrics, sizeof train_metrics,
                 "loss %.4f (best %.4f)",
                 (double)view->snap.final_train_loss,
                 (double)view->snap.best_loss);
      }
      studio_detail(ui, "Train result", train_metrics);
      if (view->snap.has_val) {
        if (view->snap.has_final_accuracy) {
          snprintf(val_metrics, sizeof val_metrics,
                   "loss %.4f / accuracy %.2f%%",
                   (double)view->snap.final_val_loss,
                   (double)view->snap.final_val_accuracy * 100.0);
        } else {
          snprintf(val_metrics, sizeof val_metrics, "loss %.4f",
                   (double)view->snap.final_val_loss);
        }
        studio_detail(ui, "Validation", val_metrics);
      }
    }
    if (view->snap.export_path[0] != '\0') {
      studio_detail(ui, "Export path", view->snap.export_path);
    } else if (state->export_path[0] != '\0') {
      studio_detail(ui, "Export path", state->export_path);
    }

    if (view->snap.error[0] != '\0') {
      (void)cui_alert(ui, "Training error",
                      .description = view->snap.error,
                      .tone = view->snap.status[0] == '\0' ||
                                      !view->snap.finished
                                  ? CUI_TONE_DANGER
                                  : CUI_TONE_WARNING,
                      .icon = CUI_ICON_TRIANGLE_ALERT);
    }
  }
}

static void studio_main(cui_ctx *ui, studio_state *state,
                        const studio_view *view)
{
  cui_column(ui, .width = CUI_FILL, .height = CUI_FILL,
             .bg = CUI_COLOR_BACKGROUND) {
    cui_row(ui, .width = CUI_FILL, .pad_x = 24, .pad_y = 16, .gap = 12,
            .align = CUI_ALIGN_CENTER) {
      cui_column(ui, .gap = 2) {
        cui_heading(ui, "Training run", 2);
        cui_secondary(ui, "Adjust parameters on the left; the charts follow "
                          "each epoch.");
      }
      cui_spacer(ui);
      if (view->running) {
        cui_badge(ui, "Training", .tone = CUI_TONE_INFO, .dot = true);
      } else if (view->snap.error[0] != '\0') {
        cui_badge(ui, "Needs attention", .tone = CUI_TONE_DANGER, .dot = true);
      } else if (view->snap.stopped) {
        cui_badge(ui, "Stopped", .tone = CUI_TONE_WARNING, .dot = true);
      } else if (view->snap.early_stopped) {
        cui_badge(ui, "Early stop", .tone = CUI_TONE_WARNING, .dot = true);
      } else if (view->snap.exported) {
        cui_badge(ui, "Complete", .tone = CUI_TONE_SUCCESS, .dot = true);
      } else {
        cui_badge(ui, "Idle", .tone = CUI_TONE_NEUTRAL, .dot = true);
      }
      if (cui_icon_button(ui,
                          state->theme == CUI_THEME_DARK ? CUI_ICON_SUN
                                                         : CUI_ICON_MOON,
                          .tooltip = "Toggle light and dark theme")) {
        state->theme = (state->theme == CUI_THEME_DARK) ? CUI_THEME_LIGHT
                                                        : CUI_THEME_DARK;
        cui_set_theme(ui, state->theme);
      }
    }
    cui_separator(ui);

    cui_scroll(ui, .id = "studio_content", .width = CUI_FILL,
               .height = CUI_FILL, .pad = 24, .gap = 18, .fade_edges = true) {
      studio_metrics(ui, view);
      studio_progress_card(ui, view);
      studio_chart(ui, "Loss", "Monitored split loss after each epoch",
                   "loss_chart", studio_draw_loss, view->history_count,
                   "%.4f", view->history_count > 0,
                   "The loss curve appears after the first epoch.");
      studio_chart(ui, "Accuracy",
                   view->snap.has_accuracy
                       ? "Share of correctly classified rows"
                       : "Reported for classification (cross-entropy) runs",
                   "accuracy_chart", studio_draw_accuracy,
                   view->history_count, "%.3f",
                   view->history_count > 0 && view->has_accuracy_history,
                   view->snap.has_accuracy
                       ? "The accuracy curve appears after the first epoch."
                       : "Accuracy is only reported for classification runs.");
      studio_run_details(ui, state, view);
    }
  }
}

static void studio_frame(cui_ctx *ui, studio_state *state)
{
  studio_columns_sync(state);

  studio_view view;
  studio_take_view(state, &view);

  const bool narrow = cui_viewport_size(ui).x < 980.0f;
  bool load_clicked = false;

  cui_row(ui, .width = CUI_FILL, .height = CUI_FILL, .gap = 0,
          .align = CUI_ALIGN_STRETCH, .bg = CUI_COLOR_BACKGROUND) {
    load_clicked = studio_sidebar(ui, state, view.running);
    if (!narrow) {
      cui_separator(ui);
    }
    studio_main(ui, state, &view);
  }

  if (load_clicked) {
    studio_load_config(state, false);
  }
}

static bool studio_parse_args(studio_state *state, int argc, char **argv,
                              bool *autostart)
{
  bool explicit_config = false;
  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "--autostart") == 0) {
      *autostart = true;
    } else if (strcmp(argv[i], "--config") == 0 && i + 1 < argc) {
      studio_copy_string(state->config_path, sizeof state->config_path,
                         argv[i + 1]);
      explicit_config = true;
      ++i;
    } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
      printf("usage: elml_studio [--config FILE] [--autostart]\n"
             "\n"
             "  --config FILE   load an .ini training configuration\n"
             "  --autostart     start training as soon as the window opens\n"
             "\n"
             "The CUI_SCREENSHOT and CUI_SCREENSHOT_FRAMES environment\n"
             "variables save a frame to PNG and quit (handy for demos).\n");
      return false;
    } else if (argv[i][0] != '-' && !explicit_config) {
      studio_copy_string(state->config_path, sizeof state->config_path,
                         argv[i]);
      explicit_config = true;
    }
  }
  if (state->config_path[0] != '\0') {
    studio_load_config(state, !explicit_config);
  }
  return true;
}

int main(int argc, char **argv)
{
  studio_state *state = (studio_state *)calloc(1u, sizeof *state);
  if (state == nullptr) {
    fprintf(stderr, "elml_studio: out of memory\n");
    return 1;
  }
  if (mtx_init(&state->mutex, mtx_plain) != thrd_success) {
    fprintf(stderr, "elml_studio: cannot create the UI mutex\n");
    free(state);
    return 1;
  }

  studio_defaults(state);
  bool autostart = false;
  if (!studio_parse_args(state, argc, argv, &autostart)) {
    mtx_destroy(&state->mutex);
    free(state);
    return 0;
  }

  cui_app *app = cui_app_create(&(cui_app_desc){
      .title = "embedlml studio",
      .width = 1320,
      .height = 880,
      .min_width = 860,
      .min_height = 600,
      .continuous = true,
      .theme = CUI_THEME_DARK,
      .accent = CUI_ACCENT_VIOLET});
  if (app == nullptr) {
    fprintf(stderr, "elml_studio: cannot create a window\n");
    mtx_destroy(&state->mutex);
    free(state);
    return 1;
  }
  state->app = app;

  bool start_requested = autostart;
  while (cui_app_frame(app)) {
    studio_frame(cui_app_ui(app), state);
    if (start_requested) {
      start_requested = false;
      studio_start(state);
    }
  }

  studio_shutdown(state);
  cui_app_destroy(app);
  studio_columns_release(&state->columns);
  mtx_destroy(&state->mutex);
  free(state);
  return 0;
}
