#define _POSIX_C_SOURCE 200809L

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

#include <time.h>

#if defined(ELML_WASM)
#include "train/memfs.h"
#else
#include <pthread.h>
#endif

#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "elml/model.h"
#include "elml/registry.h"
#include "elml/types.h"
#include "train/config.h"
#include "train/csv.h"
#include "train/evaluate.h"
#include "train/session.h"

#ifndef ELML_STUDIO_EXAMPLE_CONFIG
#define ELML_STUDIO_EXAMPLE_CONFIG ""
#endif

#define STUDIO_PATH_MAX ELML_CONFIG_PATH_MAX
#define STUDIO_LIST_MAX ELML_CONFIG_LIST_MAX
#define STUDIO_HISTORY_MAX 8192
#define STUDIO_ARG_MAX 48

#if defined(ELML_WASM)
#define STUDIO_EXPORT(name) __attribute__((export_name(#name)))
/* web/host/web_app.c calls this after the recorder boots. */
int web_app_main(void);
/* Opens an HTML file input; the page calls elml_studio_set_file() with the
 * bytes once the user picks one. */
void js_studio_pick_file(int kind)
    __attribute__((import_module("env"), import_name("js_studio_pick_file")));
#endif

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

#if defined(ELML_WASM)
typedef struct studio_mutex {
  int unused;
} studio_mutex;
#else
typedef pthread_mutex_t studio_mutex;
#endif

typedef struct studio_state {
  cui_app *app;
  cui_ctx *ui;
  studio_mutex mutex;
#if !defined(ELML_WASM)
  pthread_t worker;
#endif
  bool worker_started;

  /* Cooperative training job (wasm only): one epoch per frame. */
  elml_training web_training;
  bool web_job_active;

  bool running;
  bool finished;
  elml_training *active;
  elml_config pending;
  studio_snapshot snap;

  studio_columns columns;
  studio_columns eval_columns;

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

  /* Tabs: 0 trains, 1 evaluates a saved model against a CSV. */
  int tab;

  char eval_model_path[STUDIO_PATH_MAX];
  char eval_csv_path[STUDIO_PATH_MAX];
  char eval_inputs[STUDIO_LIST_MAX];
  char eval_targets[STUDIO_LIST_MAX];
  char eval_delimiter[8];
  bool eval_has_header;
  bool eval_score;
  bool eval_show_errors_only;
  uint32_t eval_display_limit;

  bool eval_running;       /* guarded by the mutex */
  bool eval_pending;       /* guarded by the mutex: staging holds a result */
  bool eval_seed_export;   /* guarded by the mutex: copy the exported path */
#if !defined(ELML_WASM)
  pthread_t eval_worker;
#endif
  bool eval_join_needed;

  elml_eval_result eval_result;   /* UI thread only */
  elml_eval_result eval_staging;  /* guarded by the mutex */
  bool eval_has_result;
  bool eval_ok;
  bool eval_scroll_results;

  char eval_error[1024];
  char eval_status[512];

  cui_theme_mode theme;
} studio_state;

/* Copied out of the mutex once per frame so the charts never point into the
 * worker's half-written history. */
static float studio_draw_loss[STUDIO_HISTORY_MAX];
static float studio_draw_accuracy[STUDIO_HISTORY_MAX];

/* The CUI context changes per backend: a window on the host, the page's
 * canvas instance in the browser. Every redraw request goes through here. */
static cui_ctx *studio_ui(studio_state *state)
{
  return state->ui;
}

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

#if defined(ELML_WASM)
static void studio_lock(studio_state *state)
{
  (void)state;
}

static void studio_unlock(studio_state *state)
{
  (void)state;
}
#else
static void studio_lock(studio_state *state)
{
  (void)pthread_mutex_lock(&state->mutex);
}

static void studio_unlock(studio_state *state)
{
  (void)pthread_mutex_unlock(&state->mutex);
}
#endif

static bool studio_is_running(studio_state *state)
{
  studio_lock(state);
  const bool running = state->running;
  studio_unlock(state);
  return running;
}

static bool studio_eval_running(studio_state *state)
{
  studio_lock(state);
  const bool running = state->eval_running;
  studio_unlock(state);
  return running;
}

/*
 * CUI's X11 backend wakes as soon as the display connection is readable, which
 * some drivers keep true permanently. The event-driven loop therefore redraws
 * at the refresh rate even when nothing moves, which burns a core while the
 * window sits idle. Pace the loop from the app side instead: let animations,
 * hover fades and tooltips run at full speed, give a run its epoch cadence,
 * and tick slowly when the window has nothing to show.
 */
#if !defined(ELML_WASM)
#define STUDIO_IDLE_TICK_MS 33L
#define STUDIO_RUN_TICK_MS 30L
#define STUDIO_ACTIVE_HOLD_S 0.6

static double studio_now(void)
{
  struct timespec ts;
  (void)clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

static void studio_sleep_ms(long ms)
{
  const struct timespec pause = {0, ms * 1000L * 1000L};
  (void)nanosleep(&pause, nullptr);
}

static void studio_pace(cui_ctx *ui, bool running, double *active_until)
{
  if (running) {
    studio_sleep_ms(STUDIO_RUN_TICK_MS);
    return;
  }

  const double now = studio_now();
  if (cui_needs_redraw(ui)) {
    /* Something is moving (hover, tooltip, chart tween). Keep full speed for a
     * moment afterwards so a moving pointer never waits on the slow tick. */
    *active_until = now + STUDIO_ACTIVE_HOLD_S;
    return;
  }
  if (now < *active_until) {
    return;
  }
  studio_sleep_ms(STUDIO_IDLE_TICK_MS);
}
#endif /* !ELML_WASM */

[[gnu::format(printf, 2, 3)]]
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
  state->tab = 0;
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
  state->eval_has_header = true;
  state->eval_score = true;
  state->eval_display_limit = 500u;
  snprintf(state->delimiter, sizeof state->delimiter, "%s", ",");
  snprintf(state->eval_delimiter, sizeof state->eval_delimiter, "%s", ",");
  snprintf(state->layers, sizeof state->layers, "%s", "16");
  snprintf(state->export_path, sizeof state->export_path, "%s", "model.elml");
  snprintf(state->config_path, sizeof state->config_path, "%s",
           ELML_STUDIO_EXAMPLE_CONFIG);
  snprintf(state->snap.status, sizeof state->snap.status, "%s",
           "Set data paths, then start training.");
  snprintf(state->eval_status, sizeof state->eval_status, "%s",
           "Load a model and a CSV, then evaluate.");
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

/* One column source: the CSV to scan and the two text lists it drives.
 * Training and evaluation each own one so both share the same picker. */
typedef struct studio_columns_config {
  const char *csv_path;
  const char *delimiter;
  bool has_header;
  char *inputs;
  size_t inputs_cap;
  char *outputs;
  size_t outputs_cap;
} studio_columns_config;

static studio_columns_config studio_training_columns_config(studio_state *state)
{
  return (studio_columns_config){
      .csv_path = state->csv_path,
      .delimiter = state->delimiter,
      .has_header = state->has_header,
      .inputs = state->inputs,
      .inputs_cap = sizeof state->inputs,
      .outputs = state->outputs,
      .outputs_cap = sizeof state->outputs,
  };
}

static studio_columns_config studio_eval_columns_config(studio_state *state)
{
  return (studio_columns_config){
      .csv_path = state->eval_csv_path,
      .delimiter = state->eval_delimiter,
      .has_header = state->eval_has_header,
      .inputs = state->eval_inputs,
      .inputs_cap = sizeof state->eval_inputs,
      .outputs = state->eval_targets,
      .outputs_cap = sizeof state->eval_targets,
  };
}

static bool studio_columns_signature_matches(
    const studio_columns *columns, const studio_columns_config *config)
{
  return columns->checked && strcmp(columns->path, config->csv_path) == 0 &&
         strcmp(columns->delimiter, config->delimiter) == 0 &&
         columns->has_header == config->has_header;
}

static void studio_columns_note_signature(studio_columns *columns,
                                          const studio_columns_config *config)
{
  studio_copy_string(columns->path, sizeof columns->path, config->csv_path);
  studio_copy_string(columns->delimiter, sizeof columns->delimiter,
                     config->delimiter);
  columns->has_header = config->has_header;
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

static void studio_columns_sync_lists(studio_columns *columns,
                                      const studio_columns_config *config)
{
  if (!columns->loaded) {
    return;
  }

  int input_count = 0;
  int output_count = 0;
  const bool inputs_ok = studio_columns_join(
      columns, columns->input_selected, config->inputs, config->inputs_cap,
      &input_count);
  const bool outputs_ok = studio_columns_join(
      columns, columns->output_selected, config->outputs, config->outputs_cap,
      &output_count);
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

static bool studio_columns_load_ex(studio_columns *columns,
                                   const studio_columns_config *config,
                                   bool force)
{
  if (!force && columns->loaded &&
      studio_columns_signature_matches(columns, config)) {
    return true;
  }

  if (config->csv_path[0] == '\0') {
    studio_columns_release(columns);
    snprintf(columns->status, sizeof columns->status,
             "Choose a CSV to list its columns.");
    return false;
  }

  const char delimiter = studio_delimiter_value(config->delimiter);
  if (delimiter == '\0') {
    studio_columns_release(columns);
    snprintf(columns->status, sizeof columns->status,
             "Delimiter must be one character, or \\t for a tab.");
    studio_columns_note_signature(columns, config);
    return false;
  }

  elml_csv header;
  elml_status status = elml_csv_read_header(config->csv_path, delimiter,
                                            &header);
  if (status != ELML_OK) {
    studio_columns_release(columns);
    snprintf(columns->status, sizeof columns->status,
             "Could not read columns from %.300s (%.100s).", config->csv_path,
             elml_error_string(status));
    studio_columns_note_signature(columns, config);
    return false;
  }
  if (header.cols > (uint32_t)INT_MAX) {
    elml_csv_free(&header);
    studio_columns_release(columns);
    snprintf(columns->status, sizeof columns->status,
             "The CSV header has too many columns for the picker.");
    studio_columns_note_signature(columns, config);
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
      if (config->has_header && cell != nullptr && cell[0] != '\0') {
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
    studio_columns_note_signature(columns, config);
    return false;
  }

  elml_csv_free(&header);
  columns->loaded = true;
  studio_columns_note_signature(columns, config);

  columns->unresolved_inputs = 0;
  columns->unresolved_outputs = 0;
  studio_columns_apply_list(columns, config->inputs, columns->input_selected,
                            &columns->unresolved_inputs);
  studio_columns_apply_list(columns, config->outputs, columns->output_selected,
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
  studio_columns_sync_lists(columns, config);
  return true;
}

static bool studio_columns_load(studio_state *state, bool force)
{
  const studio_columns_config config = studio_training_columns_config(state);
  return studio_columns_load_ex(&state->columns, &config, force);
}

static bool studio_eval_columns_load(studio_state *state, bool force)
{
  const studio_columns_config config = studio_eval_columns_config(state);
  return studio_columns_load_ex(&state->eval_columns, &config, force);
}

static void studio_columns_sync_ex(studio_columns *columns,
                                   const studio_columns_config *config)
{
  if (config->csv_path[0] == '\0' && !columns->loaded) {
    return;
  }
  if (studio_columns_signature_matches(columns, config)) {
    return;
  }
  (void)studio_columns_load_ex(columns, config, false);
}

static void studio_columns_sync(studio_state *state)
{
  const studio_columns_config config = studio_training_columns_config(state);
  studio_columns_sync_ex(&state->columns, &config);
}

static void studio_eval_columns_sync(studio_state *state)
{
  const studio_columns_config config = studio_eval_columns_config(state);
  const bool was_loaded = state->eval_columns.loaded;
  const bool signature_differs =
      was_loaded &&
      !studio_columns_signature_matches(&state->eval_columns, &config);
  studio_columns_sync_ex(&state->eval_columns, &config);
  if (signature_differs && state->eval_has_result) {
    elml_eval_free(&state->eval_result);
    state->eval_has_result = false;
    state->eval_ok = false;
    state->eval_error[0] = '\0';
    snprintf(state->eval_status, sizeof state->eval_status, "%s",
             "The CSV or header changed; evaluate again to refresh the "
             "results.");
  }
}

static bool studio_columns_validate_ex(studio_columns *columns,
                                       const studio_columns_config *config,
                                       bool require_selection, char *error,
                                       size_t cap)
{
  if (!studio_columns_load_ex(columns, config, false)) {
    studio_copy_string(error, cap,
                       columns->status[0] != '\0'
                           ? columns->status
                           : "Could not read the CSV header.");
    return false;
  }

  int input_count = 0;
  int output_count = 0;
  for (int i = 0; i < columns->count; ++i) {
    if (columns->input_selected[i]) {
      input_count++;
    }
    if (columns->output_selected[i]) {
      output_count++;
    }
  }
  if (require_selection && input_count == 0) {
    snprintf(error, cap, "Select at least one input column.");
    return false;
  }
  if (require_selection && output_count == 0) {
    snprintf(error, cap, "Select at least one output column.");
    return false;
  }
  const int unresolved =
      columns->unresolved_inputs + columns->unresolved_outputs;
  if (unresolved > 0) {
    snprintf(error, cap,
             "%d configured column%s not found in the CSV header. Choose "
             "them again with the picker.",
             unresolved, (unresolved == 1) ? " was" : "s were");
    return false;
  }
  for (int i = 0; i < columns->count; ++i) {
    if (columns->input_selected[i] && columns->output_selected[i]) {
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
  if (columns->has_header) {
    for (int i = 0; i < columns->count; ++i) {
      if (!columns->input_selected[i] && !columns->output_selected[i]) {
        continue;
      }
      char token[16];
      snprintf(token, sizeof token, "%d", i + 1);
      for (int j = 0; j < columns->count; ++j) {
        if (j == i || columns->names[j] == nullptr) {
          continue;
        }
        if (strcmp(columns->names[j], token) == 0) {
          snprintf(error, cap,
                   "Column %d is selected, but another header is literally "
                   "named \"%.100s\". Column numbers would be ambiguous; "
                   "rename that header.",
                   i + 1, columns->names[j]);
          return false;
        }
      }
    }
  }

  if (!studio_columns_join(columns, columns->input_selected, config->inputs,
                           config->inputs_cap, nullptr) ||
      !studio_columns_join(columns, columns->output_selected, config->outputs,
                           config->outputs_cap, nullptr)) {
    snprintf(error, cap,
             "Too many columns are selected to fit the configuration list. "
             "Choose fewer inputs or outputs.");
    return false;
  }
  return true;
}

static bool studio_columns_validate(studio_state *state, char *error,
                                    size_t cap)
{
  const studio_columns_config config = studio_training_columns_config(state);
  return studio_columns_validate_ex(&state->columns, &config, true, error, cap);
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

  /* Seed the Evaluate tab from the same file so a trained configuration can be
   * checked against its CSV without retyping the columns. */
  snprintf(state->eval_model_path, sizeof state->eval_model_path, "%s",
           state->export_path);
  snprintf(state->eval_csv_path, sizeof state->eval_csv_path, "%s",
           state->csv_path);
  snprintf(state->eval_inputs, sizeof state->eval_inputs, "%s", state->inputs);
  snprintf(state->eval_targets, sizeof state->eval_targets, "%s",
           state->outputs);
  snprintf(state->eval_delimiter, sizeof state->eval_delimiter, "%s",
           state->delimiter);
  state->eval_has_header = state->has_header;
  state->eval_score = state->outputs[0] != '\0';
  state->eval_show_errors_only = false;
  (void)studio_eval_columns_load(state, true);

  elml_eval_free(&state->eval_result);
  state->eval_has_result = false;
  state->eval_ok = false;
  state->eval_error[0] = '\0';
  snprintf(state->eval_status, sizeof state->eval_status, "%s",
           "Configuration loaded; evaluate when ready.");
}

static void studio_load_config(studio_state *state, bool quiet)
{
  if (studio_is_running(state)) {
    return;
  }
  if (studio_eval_running(state)) {
    if (!quiet) {
      studio_set_error(state,
                       "Wait for the evaluation to finish before loading "
                       "another configuration.");
    }
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

static bool studio_job_prepare(studio_state *state, elml_training *training)
{
  const elml_status status = elml_training_init(training, &state->pending);

  if (status != ELML_OK) {
    studio_lock(state);
    state->active = nullptr;
    state->running = false;
    state->finished = true;
    snprintf(state->snap.error, sizeof state->snap.error, "%s",
             (training->error[0] != '\0') ? training->error
                                          : elml_error_string(status));
    snprintf(state->snap.status, sizeof state->snap.status, "%s",
             "Could not prepare the training data.");
    studio_unlock(state);
    elml_training_destroy(training);
    return false;
  }

  studio_lock(state);
  state->active = training;
  state->snap.prepared = true;
  state->snap.train_rows = training->train.rows;
  state->snap.val_rows = training->val.rows;
  state->snap.input_dim = training->train.input_dim;
  state->snap.output_dim = training->train.output_dim;
  state->snap.has_val = training->val.rows > 0u;
  state->snap.has_accuracy = training->train.class_index != nullptr;
  state->snap.best_loss = training->best_loss;
  snprintf(state->snap.export_path, sizeof state->snap.export_path, "%s",
           training->config.export_path);

  size_t used = 0u;
  const int first = snprintf(state->snap.model, sizeof state->snap.model,
                             "%u", training->train.input_dim);
  if (first > 0) {
    used = (size_t)first;
  }
  for (uint32_t i = 0u; i < training->net.layer_count; ++i) {
    const int written = snprintf(state->snap.model + used,
                                 sizeof state->snap.model - used, " -> %u",
                                 training->net.layers[i].out_dim);
    if (written < 0 || (size_t)written >= sizeof state->snap.model - used) {
      break;
    }
    used += (size_t)written;
  }
  snprintf(state->snap.algorithm, sizeof state->snap.algorithm, "%s + %s",
           elml_loss_name(training->config.loss),
           (training->config.optimizer == ELML_OPTIMIZER_ADAM) ? "Adam" : "SGD");
  snprintf(state->snap.status, sizeof state->snap.status,
           "Training %u rows (%u validation). Stop any time to keep the best "
           "epoch.",
           training->train.rows, training->val.rows);
  studio_unlock(state);
  return true;
}

static void studio_job_finish(studio_state *state, elml_training *training,
                              elml_status status)
{
  bool exported = false;
  char error[512];
  error[0] = '\0';
  if (status == ELML_OK) {
    if (training->ran) {
      const elml_status export_status = elml_training_export(training);
      if (export_status == ELML_OK) {
        exported = true;
      } else {
        snprintf(error, sizeof error, "%s",
                 (training->error[0] != '\0') ? training->error
                                              : elml_error_string(export_status));
      }
    } else {
      snprintf(error, sizeof error, "%s",
               "Stopped before the first epoch finished; no model was "
               "exported.");
    }
  } else {
    snprintf(error, sizeof error, "%s",
             (training->error[0] != '\0') ? training->error
                                          : elml_error_string(status));
  }

  studio_lock(state);
  state->active = nullptr;
  state->running = false;
  state->finished = true;
  state->snap.finished = true;
  state->snap.exported = exported;
  state->eval_seed_export = exported;
  state->snap.early_stopped = training->early_stopped;
  state->snap.stopped = training->stopped;
  state->snap.best_epoch = training->best_epoch;
  state->snap.best_loss = training->best_loss;
  state->snap.epoch = training->epochs_done;
  state->snap.epochs = training->config.epochs;
  state->snap.has_accuracy = training->train.class_index != nullptr;
  state->snap.has_final_accuracy = training->has_final_accuracy;
  state->snap.final_train_loss = training->final_train_loss;
  state->snap.final_train_accuracy = training->final_train_accuracy;
  state->snap.final_val_loss = training->final_val_loss;
  state->snap.final_val_accuracy = training->final_val_accuracy;
  if (training->ran) {
    /* The exported model is the best-weight snapshot, so the KPI cards should
     * describe that model rather than the last epoch that ran. */
    if (training->val.rows > 0u) {
      state->snap.loss = training->final_val_loss;
      state->snap.accuracy = training->final_val_accuracy;
    } else {
      state->snap.loss = training->final_train_loss;
      state->snap.accuracy = training->final_train_accuracy;
    }
  }
  if (exported) {
    if (training->early_stopped) {
      snprintf(state->snap.status, sizeof state->snap.status,
               "Early stopped at epoch %u (best epoch %u). Exported %s.",
               training->epochs_done, training->best_epoch,
               training->config.export_path);
    } else if (training->stopped) {
      snprintf(state->snap.status, sizeof state->snap.status,
               "Stopped at epoch %u (best epoch %u). Exported %s.",
               training->epochs_done, training->best_epoch,
               training->config.export_path);
    } else {
      snprintf(state->snap.status, sizeof state->snap.status,
               "Training complete after %u epochs (best epoch %u). Exported "
               "%s.",
               training->epochs_done, training->best_epoch,
               training->config.export_path);
    }
  } else if (status == ELML_OK && !training->ran) {
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
}

#if !defined(ELML_WASM)
static void *studio_worker(void *user)
{
  studio_state *state = (studio_state *)user;
  elml_training training;
  if (!studio_job_prepare(state, &training)) {
    return NULL;
  }
  const elml_status status =
      elml_training_run(&training, studio_progress, state);
  studio_job_finish(state, &training, status);
  elml_training_destroy(&training);
  return NULL;
}
#endif

#if defined(ELML_WASM)
static void studio_web_pump(studio_state *state)
{
  if (!state->web_job_active) {
    return;
  }

  elml_train_step step;
  const elml_status status =
      elml_training_step_epoch(&state->web_training, &step);
  if (status != ELML_OK) {
    studio_job_finish(state, &state->web_training, status);
    elml_training_destroy(&state->web_training);
    state->web_job_active = false;
    return;
  }
  if (step.epoch_completed) {
    studio_progress(&step.metrics, state);
  }
  if (step.done) {
    studio_job_finish(state, &state->web_training, ELML_OK);
    elml_training_destroy(&state->web_training);
    state->web_job_active = false;
  }
  if (state->web_job_active) {
    cui_request_redraw(studio_ui(state));
  }
}
#endif

static void studio_start(studio_state *state)
{
  if (studio_is_running(state)) {
    return;
  }
#if !defined(ELML_WASM)
  if (state->worker_started) {
    (void)pthread_join(state->worker, nullptr);
    state->worker_started = false;
  }
#endif

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

  /* Wake the event-driven loop for the first snapshot. */
  cui_request_redraw(studio_ui(state));

#if defined(ELML_WASM)
  /* No threads in the browser: prepare here and run one epoch per frame in
   * studio_web_pump(), which the page's frame callback drives. */
  if (!studio_job_prepare(state, &state->web_training)) {
    return; /* prepare published the error and cleared the running flag */
  }
  state->web_job_active = true;
#else
  if (pthread_create(&state->worker, nullptr, studio_worker, state) != 0) {
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
#endif
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
#if defined(ELML_WASM)
  if (state->web_job_active) {
    elml_training_destroy(&state->web_training);
    state->web_job_active = false;
  }
#else
  if (state->worker_started) {
    (void)pthread_join(state->worker, nullptr);
    state->worker_started = false;
  }
  if (state->eval_join_needed) {
    (void)pthread_join(state->eval_worker, nullptr);
    state->eval_join_needed = false;
  }
#endif
  elml_eval_free(&state->eval_result);
  elml_eval_free(&state->eval_staging);
}

/* ------------------------------------------------------------------------- */
/* Evaluation worker                                                         */
/* ------------------------------------------------------------------------- */

[[gnu::format(printf, 3, 4)]]
static void studio_eval_append(char *out, size_t cap, const char *format, ...)
{
  const size_t used = strlen(out);
  if (used + 1u >= cap) {
    return;
  }
  va_list args;
  va_start(args, format);
  (void)vsnprintf(out + used, cap - used, format, args);
  va_end(args);
}

static void studio_eval_summary(const elml_eval_result *result, char *out,
                                size_t cap)
{
  if (result == nullptr || out == nullptr || cap == 0u) {
    return;
  }
  if (result->row_count == 0u) {
    snprintf(out, cap, "%s", "No rows were evaluated.");
  } else if (result->scored && result->classification) {
    const double accuracy =
        (result->scored_rows == 0u)
            ? 0.0
            : (double)result->correct / (double)result->scored_rows;
    snprintf(out, cap, "%u rows checked: %llu correct (%.1f%%)",
             result->row_count, (unsigned long long)result->correct,
             accuracy * 100.0);
  } else if (result->scored) {
    snprintf(out, cap, "%u rows checked: MSE %.6f, MAE %.6f",
             result->row_count, result->mse, result->mae);
  } else {
    snprintf(out, cap, "%u rows predicted (scoring is off).",
             result->row_count);
  }
  if (result->skipped > 0u) {
    studio_eval_append(out, cap, "; %llu skipped",
                       (unsigned long long)result->skipped);
  }
}

static void studio_eval_execute(studio_state *state)
{
  const elml_eval_options options = {
      .model_path = state->eval_model_path,
      .csv_path = state->eval_csv_path,
      .delimiter = studio_delimiter_value(state->eval_delimiter),
      .has_header = state->eval_has_header,
      .inputs = state->eval_inputs,
      .targets = state->eval_targets,
      .score = state->eval_score,
      .skip_bad_rows = true,
      .limit = 0u,
  };

  elml_eval_result result;
  const elml_status status = elml_eval_run(&options, &result);

  studio_lock(state);
  /* Move ownership of every allocation to the UI thread. */
  state->eval_staging = result;
  state->eval_pending = true;
  state->eval_running = false;
  state->eval_ok = status == ELML_OK;
  if (status == ELML_OK) {
    state->eval_error[0] = '\0';
    studio_eval_summary(&result, state->eval_status,
                        sizeof state->eval_status);
  } else {
    snprintf(state->eval_error, sizeof state->eval_error, "%s",
             (result.message[0] != '\0') ? result.message
                                         : elml_error_string(status));
    snprintf(state->eval_status, sizeof state->eval_status, "%s",
             "Evaluation failed.");
  }
  studio_unlock(state);

  cui_request_redraw(studio_ui(state));
}

#if !defined(ELML_WASM)
static void *studio_eval_worker(void *user)
{
  studio_eval_execute((studio_state *)user);
  return NULL;
}
#endif

static void studio_eval_start(studio_state *state)
{
  if (studio_eval_running(state)) {
    return;
  }
#if !defined(ELML_WASM)
  if (state->eval_join_needed) {
    (void)pthread_join(state->eval_worker, nullptr);
    state->eval_join_needed = false;
  }
#endif

  if (state->eval_model_path[0] == '\0') {
    snprintf(state->eval_error, sizeof state->eval_error, "%s",
             "Choose the saved model (.elml) to evaluate.");
    return;
  }
  if (state->eval_csv_path[0] == '\0') {
    snprintf(state->eval_error, sizeof state->eval_error, "%s",
             "Choose the CSV file with the rows to check.");
    return;
  }
  if (studio_delimiter_value(state->eval_delimiter) == '\0') {
    snprintf(state->eval_error, sizeof state->eval_error, "%s",
             "Delimiter must be one character, or \\t for a tab.");
    return;
  }

  char column_error[320];
  const studio_columns_config columns_config =
      studio_eval_columns_config(state);
  if (!studio_columns_validate_ex(&state->eval_columns, &columns_config, false,
                                  column_error, sizeof column_error)) {
    snprintf(state->eval_error, sizeof state->eval_error, "%.1000s",
             column_error);
    return;
  }

  studio_lock(state);
  state->eval_running = true;
  state->eval_ok = false;
  state->eval_scroll_results = false;
  state->eval_error[0] = '\0';
  snprintf(state->eval_status, sizeof state->eval_status,
           "Evaluating every row against the model...");
  studio_unlock(state);

  cui_request_redraw(studio_ui(state));

#if defined(ELML_WASM)
  /* No threads in the browser: run the evaluation in this frame. */
  studio_eval_execute(state);
#else
  if (pthread_create(&state->eval_worker, nullptr, studio_eval_worker, state) !=
      0) {
    studio_lock(state);
    state->eval_running = false;
    snprintf(state->eval_error, sizeof state->eval_error, "%s",
             "Could not create the evaluation thread.");
    studio_unlock(state);
    return;
  }
  state->eval_join_needed = true;
#endif
}

static void studio_eval_collect(studio_state *state)
{
  char seeded[STUDIO_PATH_MAX];
  seeded[0] = '\0';
  bool pending = false;
  elml_eval_result incoming;
  memset(&incoming, 0, sizeof incoming);

  studio_lock(state);
  if (state->eval_seed_export && !state->eval_running) {
    state->eval_seed_export = false;
    snprintf(seeded, sizeof seeded, "%s", state->snap.export_path);
  }
  if (state->eval_pending) {
    incoming = state->eval_staging;
    memset(&state->eval_staging, 0, sizeof state->eval_staging);
    state->eval_pending = false;
    pending = true;
  }
  studio_unlock(state);

  if (seeded[0] != '\0') {
    snprintf(state->eval_model_path, sizeof state->eval_model_path, "%s",
             seeded);
    if (!pending) {
      elml_eval_free(&state->eval_result);
      state->eval_has_result = false;
      state->eval_ok = false;
      state->eval_error[0] = '\0';
      snprintf(state->eval_status, sizeof state->eval_status,
               "A new model was exported; evaluate it to check the CSV.");
    }
  }

  if (pending) {
    elml_eval_free(&state->eval_result);
    state->eval_result = incoming;
    state->eval_has_result = true;
    state->eval_scroll_results = state->eval_ok;
  }

#if !defined(ELML_WASM)
  if (state->eval_join_needed) {
    (void)pthread_join(state->eval_worker, nullptr);
    state->eval_join_needed = false;
  }
#endif
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

static bool studio_pick_file(studio_state *state, const char *id,
                             const cui_file_dialog_opts *opts, char *out,
                             size_t cap)
{
#if defined(ELML_WASM)
  /*
   * The page owns the file picker. We only tell it which slot was clicked;
   * when the user chooses a file, JS registers the bytes and calls
   * elml_studio_set_file() with the matching kind.
   */
  (void)state;
  (void)opts;
  (void)out;
  (void)cap;
  int kind = -1;
  if (strcmp(id, "config_path") == 0 ||
      strcmp(id, "eval_config_path") == 0) {
    kind = 0;
  } else if (strcmp(id, "csv_path") == 0) {
    kind = 1;
  } else if (strcmp(id, "val_csv_path") == 0) {
    kind = 2;
  } else if (strcmp(id, "eval_model_path") == 0) {
    kind = 3;
  } else if (strcmp(id, "eval_csv_path") == 0) {
    kind = 4;
  }
  if (kind >= 0) {
    js_studio_pick_file(kind);
  }
  return false;
#else
  (void)id;
  return cui_app_file_dialog(state->app, opts, out, cap);
#endif
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
      if (studio_pick_file(
              state, id,
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

typedef struct studio_picker_text {
  const char *id_prefix;
  const char *inputs_label;
  const char *outputs_label;
  const char *inputs_placeholder;
  const char *outputs_placeholder;
  const char *inputs_tooltip;
  const char *outputs_tooltip;
  const char *inputs_text_hint;
  const char *outputs_text_hint;
  const char *empty_hint;
  bool allow_text;
} studio_picker_text;

static bool studio_column_picker_ex(cui_ctx *ui, studio_columns *columns,
                                    const studio_columns_config *config,
                                    bool running,
                                    const studio_picker_text *text)
{
  if (columns->loaded &&
      !studio_columns_signature_matches(columns, config)) {
    cui_muted(ui, "Reading the CSV header...");
    return false;
  }

  if (!columns->loaded) {
    if (text->allow_text) {
      char input_id[64];
      char output_id[64];
      snprintf(input_id, sizeof input_id, "%s_inputs_text", text->id_prefix);
      snprintf(output_id, sizeof output_id, "%s_outputs_text",
               text->id_prefix);
      studio_text_field(ui, input_id, text->inputs_label, config->inputs,
                        config->inputs_cap, "sepal_length,sepal_width,...",
                        text->inputs_text_hint, text->inputs_tooltip, running);
      studio_text_field(ui, output_id, text->outputs_label, config->outputs,
                        config->outputs_cap, "species",
                        text->outputs_text_hint, text->outputs_tooltip,
                        running);
    } else {
      cui_muted(ui, text->empty_hint);
    }
    if (config->csv_path[0] != '\0' &&
        cui_button(ui, "Read CSV columns", .icon = CUI_ICON_REFRESH_CW,
                   .variant = CUI_BUTTON_GHOST, .size = CUI_SIZE_SM,
                   .tooltip = "Read the first CSV row and populate the column "
                              "pickers",
                   .disabled = running)) {
      (void)studio_columns_load_ex(columns, config, true);
    }
    if (columns->status[0] != '\0') {
      cui_muted(ui, columns->status);
    }
    return false;
  }

  const bool searchable = columns->count > 8;
  bool inputs_changed = false;
  bool outputs_changed = false;
  bool reloaded = false;

  char inputs_id[64];
  char outputs_id[64];
  snprintf(inputs_id, sizeof inputs_id, "%s_inputs", text->id_prefix);
  snprintf(outputs_id, sizeof outputs_id, "%s_outputs", text->id_prefix);

  cui_disabled_scope(ui, running) {
    inputs_changed = cui_multi_select_ex(
        ui, inputs_id, (const char *const *)columns->labels, columns->count,
        columns->input_selected,
        &(cui_multi_select_opts){.label = text->inputs_label,
                                 .placeholder = text->inputs_placeholder,
                                 .searchable = searchable,
                                 .max_chips = 2,
                                 .width = CUI_FILL});
    cui_tooltip(ui, text->inputs_tooltip);

    outputs_changed = cui_multi_select_ex(
        ui, outputs_id, (const char *const *)columns->labels, columns->count,
        columns->output_selected,
        &(cui_multi_select_opts){.label = text->outputs_label,
                                 .placeholder = text->outputs_placeholder,
                                 .searchable = searchable,
                                 .max_chips = 2,
                                 .width = CUI_FILL});
    cui_tooltip(ui, text->outputs_tooltip);
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
    studio_columns_sync_lists(columns, config);
  }
  memcpy(columns->input_previous, columns->input_selected,
         (size_t)columns->count * sizeof(bool));
  memcpy(columns->output_previous, columns->output_selected,
         (size_t)columns->count * sizeof(bool));

  if (cui_button(ui, "Rescan CSV columns", .icon = CUI_ICON_REFRESH_CW,
                 .variant = CUI_BUTTON_GHOST, .size = CUI_SIZE_SM,
                 .tooltip = "Re-read the CSV header after editing the file",
                 .disabled = running)) {
    reloaded = studio_columns_load_ex(columns, config, true);
  }
  if (columns->status[0] != '\0') {
    cui_muted(ui, columns->status);
  }
  return inputs_changed || outputs_changed || reloaded;
}

static void studio_column_picker(cui_ctx *ui, studio_state *state,
                                 bool running)
{
  const studio_columns_config config = studio_training_columns_config(state);
  const studio_picker_text text = {
      .id_prefix = "column",
      .inputs_label = "Input columns",
      .outputs_label = "Output column(s)",
      .inputs_placeholder = "Choose one or more inputs",
      .outputs_placeholder = "Choose the target columns",
      .inputs_tooltip = "Feature columns for the model. Choosing one here "
                        "removes it from the outputs.",
      .outputs_tooltip = "Target columns for the model. Choosing one here "
                         "removes it from the inputs.",
      .inputs_text_hint = "Column names are matched against the CSV header. "
                          "The picker appears once the file can be read.",
      .outputs_text_hint = "One name for classification, or a comma-separated "
                           "list for regression.",
      .empty_hint = "Choose a training CSV to list its columns.",
      .allow_text = true,
  };
  (void)studio_column_picker_ex(ui, &state->columns, &config, running, &text);
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
          if (studio_pick_file(
                  state, "config_path",
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
      cui_text(ui, "Optimizer", .size = 12,
               .color = CUI_COLOR_TEXT_SECONDARY);
      cui_disabled_scope(ui, running) {
        (void)cui_segmented(ui, "optimizer", studio_optimizer_labels,
                            STUDIO_COUNT(studio_optimizer_labels),
                            &state->optimizer_index, .full_width = true,
                            .size = CUI_SIZE_SM);
      }
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
  char loss_delta_label[64] = "";
  char accuracy_delta_label[64] = "";
  float loss_delta = 0.0f;
  float accuracy_delta = 0.0f;
  bool has_loss_delta = false;
  bool has_accuracy_delta = false;

  if (view->snap.epoch > 0u) {
    snprintf(epoch_value, sizeof epoch_value, "%u", view->snap.epoch);
  } else {
    snprintf(epoch_value, sizeof epoch_value, "%s", "-");
  }
  if (view->snap.epoch > 0u) {
    snprintf(loss_value, sizeof loss_value, "%.3f",
             (double)view->snap.loss);
  } else {
    snprintf(loss_value, sizeof loss_value, "%s", "-");
  }
  if (view->snap.epoch > 0u && view->snap.has_accuracy) {
    snprintf(accuracy_value, sizeof accuracy_value, "%.1f%%",
             (double)view->snap.accuracy * 100.0);
  } else {
    snprintf(accuracy_value, sizeof accuracy_value, "%s", "-");
  }
  if (view->snap.best_epoch > 0u) {
    snprintf(best_value, sizeof best_value, "%u", view->snap.best_epoch);
  } else {
    snprintf(best_value, sizeof best_value, "%s", "-");
  }

  if (view->history_count > 1) {
    const int last = view->history_count - 1;
    const float previous_loss = studio_draw_loss[last - 1];
    const float current_loss = studio_draw_loss[last];
    if (previous_loss > 0.0f) {
      loss_delta = (current_loss - previous_loss) / previous_loss * 100.0f;
      snprintf(loss_delta_label, sizeof loss_delta_label, "%+.1f%%",
               (double)loss_delta);
      has_loss_delta = true;
    }
    if (view->has_accuracy_history) {
      const float previous_accuracy = studio_draw_accuracy[last - 1];
      const float current_accuracy = studio_draw_accuracy[last];
      accuracy_delta = (current_accuracy - previous_accuracy) * 100.0f;
      snprintf(accuracy_delta_label, sizeof accuracy_delta_label, "%+.2f pp",
               (double)accuracy_delta);
      has_accuracy_delta = true;
    }
  }

  const int spark_count = view->history_count;
  const float *loss_spark = (spark_count > 1) ? studio_draw_loss : nullptr;
  const float *accuracy_spark =
      (spark_count > 1 && view->has_accuracy_history) ? studio_draw_accuracy
                                                     : nullptr;

  cui_box(ui, .dir = CUI_GRID, .columns = 4, .min_column_width = 150,
          .gap = 12) {
    cui_metric(ui, "Epoch", epoch_value, .icon = CUI_ICON_ACTIVITY);
    cui_metric(ui, "Val loss", loss_value,
               .icon = CUI_ICON_TRENDING_DOWN,
               .spark = loss_spark, .spark_count = spark_count,
               .lower_is_better = true,
               .delta = has_loss_delta ? loss_delta : 0.0f,
               .delta_label = has_loss_delta ? loss_delta_label : nullptr);
    cui_metric(ui, "Accuracy", accuracy_value, .icon = CUI_ICON_TARGET,
               .spark = accuracy_spark,
               .spark_count = accuracy_spark != nullptr ? spark_count : 0,
               .delta = has_accuracy_delta ? accuracy_delta : 0.0f,
               .delta_label =
                   has_accuracy_delta ? accuracy_delta_label : nullptr);
    cui_metric(ui, "Best epoch", best_value, .icon = CUI_ICON_ZAP);
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
    cui_row(ui, .width = CUI_FILL, .align = CUI_ALIGN_CENTER, .gap = 18) {
      cui_column(ui, .width = CUI_FILL, .gap = 10) {
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
      if (view->snap.epochs > 0u) {
        cui_progress_ring(ui, "Done", fraction, 74.0f);
      }
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

static void studio_data_split(cui_ctx *ui, const studio_view *view)
{
  const uint32_t train_rows = view->snap.train_rows;
  const uint32_t val_rows = view->snap.val_rows;
  const uint32_t total_rows = train_rows + val_rows;

  cui_card(ui, .width = CUI_FILL, .gap = 12) {
    cui_row(ui, .width = CUI_FILL, .align = CUI_ALIGN_CENTER) {
      cui_text(ui, "Data split", .font = CUI_FONT_MEDIUM, .size = 15);
      cui_spacer(ui);
      if (total_rows > 0u) {
        char ratio[40];
        const double percent =
            (double)train_rows / (double)total_rows * 100.0;
        snprintf(ratio, sizeof ratio, "%.0f%% train", percent);
        cui_badge(ui, ratio, .tone = CUI_TONE_NEUTRAL,
                  .size = CUI_SIZE_SM);
      }
    }

    if (total_rows == 0u) {
      cui_text(ui, "Train and validation rows appear after the data loads.",
               .size = 12, .color = CUI_COLOR_TEXT_MUTED);
    } else {
      float values[2] = {(float)train_rows, (float)val_rows};
      const char *labels[2] = {"Train", "Validation"};
      cui_donut_chart(ui, "data_split_chart", values, labels, 2,
                      .height = 150.0f, .width = CUI_FILL);
      char rows[128];
      snprintf(rows, sizeof rows, "%u train / %u validation rows",
               train_rows, val_rows);
      cui_muted(ui, rows);
      if (view->snap.input_dim > 0u && view->snap.output_dim > 0u) {
        char shape[64];
        snprintf(shape, sizeof shape, "%u inputs to %u outputs",
                 view->snap.input_dim, view->snap.output_dim);
        cui_muted(ui, shape);
      }
    }
  }
}

static void studio_log_item(cui_ctx *ui, int index, int count,
                            const char *title, const char *time,
                            const char *description, const char *icon,
                            cui_tone tone)
{
  cui_timeline_item(ui, title, .time = time, .description = description,
                    .icon = icon, .tone = tone, .last = index == count - 1);
}

static void studio_run_log(cui_ctx *ui, const studio_view *view)
{
  const studio_snapshot *snap = &view->snap;
  int count = 1; /* configuration is always present */
  if (snap->prepared) {
    count++;
  }
  if (snap->epoch > 0u) {
    count++;
  }
  if (snap->best_epoch > 0u) {
    count++;
  }
  if (snap->early_stopped || snap->stopped) {
    count++;
  }
  if (snap->exported) {
    count++;
  }
  if (snap->error[0] != '\0') {
    count++;
  }

  cui_card(ui, .width = CUI_FILL, .gap = 14) {
    cui_row(ui, .width = CUI_FILL, .align = CUI_ALIGN_CENTER) {
      cui_text(ui, "Run log", .font = CUI_FONT_MEDIUM, .size = 15);
      cui_spacer(ui);
      if (snap->error[0] != '\0') {
        cui_badge(ui, "Error", .tone = CUI_TONE_DANGER, .dot = true);
      } else if (snap->exported) {
        cui_badge(ui, "Exported", .tone = CUI_TONE_SUCCESS, .dot = true);
      } else if (snap->early_stopped || snap->stopped) {
        cui_badge(ui, "Stopped", .tone = CUI_TONE_WARNING, .dot = true);
      } else if (view->running) {
        cui_badge(ui, "Training", .tone = CUI_TONE_INFO, .dot = true);
      } else {
        cui_badge(ui, "Ready", .tone = CUI_TONE_NEUTRAL, .dot = true);
      }
    }

    int index = 0;
    if (snap->error[0] != '\0') {
      studio_log_item(ui, index++, count, "Run failed", nullptr,
                      "See the error details below.", CUI_ICON_TRIANGLE_ALERT,
                      CUI_TONE_DANGER);
    }
    if (snap->exported) {
      studio_log_item(ui, index++, count, "Model exported", "done",
                      snap->export_path[0] != '\0' ? snap->export_path
                                                   : "Weights written",
                      CUI_ICON_DOWNLOAD, CUI_TONE_SUCCESS);
    }
    if (snap->early_stopped || snap->stopped) {
      char time[32];
      if (snap->best_epoch > 0u) {
        snprintf(time, sizeof time, "epoch %u", snap->best_epoch);
      } else {
        snprintf(time, sizeof time, "%s", "stopped");
      }
      studio_log_item(ui, index++, count,
                      snap->early_stopped ? "Early stopping" : "Stopped",
                      time,
                      snap->early_stopped
                          ? "No validation improvement inside the patience "
                            "window."
                          : "Training stopped; the best epoch was restored.",
                      CUI_ICON_SQUARE, CUI_TONE_WARNING);
    }
    if (snap->best_epoch > 0u) {
      char time[32];
      char description[96];
      snprintf(time, sizeof time, "epoch %u", snap->best_epoch);
      snprintf(description, sizeof description, "Validation loss %.4f",
               (double)snap->best_loss);
      studio_log_item(ui, index++, count, "Best validation", time,
                      description, CUI_ICON_STAR, CUI_TONE_ACCENT);
    }
    if (snap->epoch > 0u) {
      char time[48];
      char description[160];
      snprintf(time, sizeof time, "%u / %u", snap->epoch, snap->epochs);
      if (snap->has_accuracy) {
        snprintf(description, sizeof description,
                 "loss %.4f, accuracy %.2f%%", (double)snap->loss,
                 (double)snap->accuracy * 100.0);
      } else {
        snprintf(description, sizeof description, "loss %.4f",
                 (double)snap->loss);
      }
      studio_log_item(ui, index++, count, "Training", time, description,
                      CUI_ICON_ACTIVITY, CUI_TONE_INFO);
    }
    if (snap->prepared) {
      char description[128];
      snprintf(description, sizeof description,
               "%u train / %u validation rows", snap->train_rows,
               snap->val_rows);
      studio_log_item(ui, index++, count, "Data ready", nullptr, description,
                      CUI_ICON_DATABASE, CUI_TONE_SUCCESS);
    }
    studio_log_item(ui, index++, count, "Configuration ready", nullptr,
                    snap->status[0] != '\0' ? snap->status
                                            : "Settings loaded.",
                    CUI_ICON_CHECK, CUI_TONE_NEUTRAL);
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

/* ------------------------------------------------------------------------- */
/* Evaluate tab                                                              */
/* ------------------------------------------------------------------------- */

static void studio_eval_vector_text(const float *values, uint32_t count,
                                    char *out, size_t cap)
{
  if (out == nullptr || cap == 0u) {
    return;
  }
  if (values == nullptr) {
    snprintf(out, cap, "%s", "-");
    return;
  }
  out[0] = '\0';
  size_t used = 0u;
  for (uint32_t i = 0u; i < count; ++i) {
    const int written = snprintf(out + used, cap - used, "%s%.4f",
                                 (i == 0u) ? "" : ", ", (double)values[i]);
    if (written < 0 || (size_t)written >= cap - used) {
      out[cap - 1u] = '\0';
      return;
    }
    used += (size_t)written;
  }
}

static void studio_eval_error_text(const elml_eval_result *result,
                                   const elml_eval_row *row, char *out,
                                   size_t cap)
{
  const uint32_t index = (uint32_t)(row - result->rows);
  const float *values = elml_eval_values(result, index);
  const float *targets = elml_eval_targets(result, index);
  if (out == nullptr || cap == 0u) {
    return;
  }
  if (values == nullptr || targets == nullptr) {
    snprintf(out, cap, "%s", "-");
    return;
  }
  out[0] = '\0';
  size_t used = 0u;
  for (uint32_t i = 0u; i < result->output_dim; ++i) {
    const int written =
        snprintf(out + used, cap - used, "%s%.4f", (i == 0u) ? "" : ", ",
                 fabs((double)values[i] - (double)targets[i]));
    if (written < 0 || (size_t)written >= cap - used) {
      out[cap - 1u] = '\0';
      return;
    }
    used += (size_t)written;
  }
}

static void studio_eval_save(studio_state *state)
{
  if (!state->eval_has_result) {
    return;
  }
  char picked[STUDIO_PATH_MAX];
#if defined(ELML_WASM)
  /* Browser: write into the in-memory table; the Files panel downloads it. */
  snprintf(picked, sizeof picked, "%s", "predictions.csv");
#else
  picked[0] = '\0';
  if (!cui_app_file_dialog(
          state->app,
          &(cui_file_dialog_opts){.kind = CUI_FILE_SAVE,
                                  .title = "Save predictions",
                                  .filter_name = "CSV files",
                                  .filter = "csv",
                                  .default_name = "predictions.csv"},
          picked, sizeof picked)) {
    return;
  }
#endif

  const elml_status status = elml_eval_write_csv(&state->eval_result, picked);
  if (status == ELML_OK) {
    state->eval_error[0] = '\0';
    snprintf(state->eval_status, sizeof state->eval_status,
             "Saved %u rows with predictions to %.360s",
             state->eval_result.row_count, picked);
  } else {
    snprintf(state->eval_error, sizeof state->eval_error,
             "Could not write %.800s (%.100s).", picked,
             elml_error_string(status));
  }
}

static bool studio_eval_controls(cui_ctx *ui, studio_state *state)
{
  const bool running = studio_eval_running(state);
  bool load_config = false;

  cui_card(ui, .width = CUI_FILL, .gap = 12) {
    cui_row(ui, .width = CUI_FILL, .align = CUI_ALIGN_CENTER) {
      cui_column(ui, .gap = 2) {
        cui_text(ui, "Check a saved model", .font = CUI_FONT_MEDIUM,
                 .size = 15);
        cui_secondary(ui, "Runs every row through the exported model and "
                          "reports what it predicted.");
      }
      cui_spacer(ui);
      if (running) {
        cui_badge(ui, "Evaluating", .tone = CUI_TONE_INFO, .dot = true);
      } else if (state->eval_error[0] != '\0') {
        cui_badge(ui, "Needs attention", .tone = CUI_TONE_DANGER, .dot = true);
      } else if (state->eval_has_result) {
        cui_badge(ui, "Complete", .tone = CUI_TONE_SUCCESS, .dot = true);
      } else {
        cui_badge(ui, "Idle", .tone = CUI_TONE_NEUTRAL, .dot = true);
      }
    }

    cui_row(ui, .width = CUI_FILL, .gap = 8, .align = CUI_ALIGN_END) {
      (void)cui_input(ui, "eval_config_path", state->config_path,
                      sizeof state->config_path,
                      .label = "Config file (.ini, optional)",
                      .width = CUI_FILL, .disabled = running);
      cui_tooltip(ui, "Load the same .ini used for training to fill the model, "
                      "CSV, columns, header and delimiter below.");
      if (cui_icon_button(ui, CUI_ICON_FOLDER_OPEN, .disabled = running,
                          .tooltip = "Browse for an .ini file")) {
        char picked[STUDIO_PATH_MAX];
        picked[0] = '\0';
        if (studio_pick_file(
                state, "eval_config_path",
                &(cui_file_dialog_opts){.kind = CUI_FILE_OPEN,
                                        .title = "Open training configuration",
                                        .filter_name = "INI files",
                                        .filter = "ini"},
                picked, sizeof picked)) {
          snprintf(state->config_path, sizeof state->config_path, "%s",
                   picked);
        }
      }
      if (cui_button(ui, "Load", .icon = CUI_ICON_REFRESH_CW,
                     .variant = CUI_BUTTON_GHOST, .size = CUI_SIZE_SM,
                     .disabled = running,
                     .tooltip = "Fill the fields below from this .ini")) {
        if (state->config_path[0] == '\0') {
          snprintf(state->eval_error, sizeof state->eval_error, "%s",
                   "Choose an .ini configuration file first.");
        } else {
          load_config = true;
        }
      }
    }

    studio_path_field(ui, state, "eval_model_path", "Saved model (.elml)",
                      state->eval_model_path, sizeof state->eval_model_path,
                      CUI_FILE_OPEN, "embedlml models", "elml",
                      "The model exported by training, or any other .elml.",
                      running);
    studio_path_field(ui, state, "eval_csv_path", "CSV to check",
                      state->eval_csv_path, sizeof state->eval_csv_path,
                      CUI_FILE_OPEN, "CSV files", "csv",
                      "Every data row in this file is passed through the "
                      "model. The training CSV works when the model knows its "
                      "columns.",
                      running);

    cui_row(ui, .width = CUI_FILL, .gap = 12, .align = CUI_ALIGN_END,
            .wrap = true) {
      if (cui_button(ui, "Use training data & export",
                     .icon = CUI_ICON_REFRESH_CW, .variant = CUI_BUTTON_GHOST,
                     .size = CUI_SIZE_SM, .disabled = running,
                     .tooltip = "Copy the training CSV, exported model path "
                                "and column selection into this tab")) {
        if (state->snap.export_path[0] != '\0') {
          snprintf(state->eval_model_path, sizeof state->eval_model_path, "%s",
                   state->snap.export_path);
        } else if (state->export_path[0] != '\0') {
          snprintf(state->eval_model_path, sizeof state->eval_model_path, "%s",
                   state->export_path);
        }
        snprintf(state->eval_csv_path, sizeof state->eval_csv_path, "%s",
                 state->csv_path);
        snprintf(state->eval_inputs, sizeof state->eval_inputs, "%s",
                 state->inputs);
        snprintf(state->eval_targets, sizeof state->eval_targets, "%s",
                 state->outputs);
        snprintf(state->eval_delimiter, sizeof state->eval_delimiter, "%s",
                 state->delimiter);
        state->eval_has_header = state->has_header;
        state->eval_score = true;
        state->eval_show_errors_only = false;
        (void)studio_eval_columns_load(state, true);
        elml_eval_free(&state->eval_result);
        state->eval_has_result = false;
        state->eval_ok = false;
        state->eval_error[0] = '\0';
        snprintf(state->eval_status, sizeof state->eval_status, "%s",
                 "Training data selected; evaluate when ready.");
      }
      cui_spacer(ui);
      (void)cui_input(ui, "eval_delimiter", state->eval_delimiter,
                      sizeof state->eval_delimiter, .label = "Delimiter",
                      .width = CUI_PX(110), .disabled = running);
      cui_tooltip(ui, "Field separator in the evaluation CSV: one character, "
                      "or \\t for a tab.");
    }

    cui_row(ui, .width = CUI_FILL, .gap = 18, .wrap = true) {
      (void)cui_switch(ui, "CSV has a header", &state->eval_has_header,
                       .disabled = running);
      cui_tooltip(ui, "Skip the first row when reading the evaluation CSV.");
      (void)cui_switch(ui, "Score against targets", &state->eval_score,
                       .disabled = running);
      cui_tooltip(ui, "Compare each prediction with the target column to count "
                      "correct rows or regression error. Turn it off to only "
                      "list predictions.");
    }

    const studio_columns_config columns_config =
        studio_eval_columns_config(state);
    const studio_picker_text column_text = {
        .id_prefix = "eval_column",
        .inputs_label = "Input columns",
        .outputs_label = "Target column(s)",
        .inputs_placeholder = "Choose one or more inputs",
        .outputs_placeholder = "Choose the target columns",
        .inputs_tooltip = "Feature columns in the evaluation CSV, in model "
                          "input order. Choosing one here removes it from the "
                          "targets.",
        .outputs_tooltip = "Columns compared with the predictions. Choosing "
                           "one here removes it from the inputs.",
        .inputs_text_hint = "",
        .outputs_text_hint = "",
        .empty_hint = "Choose the CSV to check to list its columns. Leaving "
                      "both empty uses the columns stored in the model.",
        .allow_text = false,
    };
    if (studio_column_picker_ex(ui, &state->eval_columns, &columns_config,
                                running, &column_text) &&
        state->eval_has_result) {
      elml_eval_free(&state->eval_result);
      state->eval_has_result = false;
      state->eval_ok = false;
      state->eval_error[0] = '\0';
      snprintf(state->eval_status, sizeof state->eval_status, "%s",
               "Columns changed; evaluate again to refresh the results.");
    }

    cui_row(ui, .width = CUI_FILL, .gap = 10, .align = CUI_ALIGN_CENTER,
            .wrap = true) {
      if (cui_button(ui, running ? "Evaluating..." : "Evaluate CSV",
                     .variant = CUI_BUTTON_PRIMARY,
                     .icon = running ? CUI_ICON_LOADER_CIRCLE : CUI_ICON_PLAY,
                     .loading = running, .disabled = running,
                     .tooltip = "Run every row through the model in the "
                                "background")) {
        studio_eval_start(state);
      }
      if (state->eval_has_result && !running) {
        if (cui_button(ui, "Save predictions", .variant = CUI_BUTTON_OUTLINE,
                       .icon = CUI_ICON_SAVE, .size = CUI_SIZE_SM,
                       .tooltip = "Write the CSV columns plus the predictions "
                                  "to a new file")) {
          studio_eval_save(state);
        }
      }
      cui_spacer(ui);
      cui_muted(ui, "Evaluation runs on a worker thread.");
    }

    if (running) {
      cui_progress(ui, 0.0f, .indeterminate = true, .tone = CUI_TONE_INFO);
    }
    if (state->eval_error[0] != '\0') {
      (void)cui_alert(ui, "Evaluation error",
                      .description = state->eval_error, .tone = CUI_TONE_DANGER,
                      .icon = CUI_ICON_TRIANGLE_ALERT);
    } else {
      cui_muted(ui, state->eval_status);
    }
  }

  return load_config;
}

static void studio_eval_metrics(cui_ctx *ui, const studio_state *state)
{
  const elml_eval_result *result = &state->eval_result;
  char rows[32];
  char skipped[32];
  snprintf(rows, sizeof rows, "%u", result->row_count);
  snprintf(skipped, sizeof skipped, "%llu",
           (unsigned long long)result->skipped);

  if (result->scored && result->classification) {
    char accuracy[32];
    char correct[48];
    const double value =
        (result->scored_rows == 0u)
            ? 0.0
            : (double)result->correct / (double)result->scored_rows;
    snprintf(accuracy, sizeof accuracy, "%.1f%%", value * 100.0);
    snprintf(correct, sizeof correct, "%llu / %llu",
             (unsigned long long)result->correct,
             (unsigned long long)result->scored_rows);
    cui_box(ui, .dir = CUI_GRID, .columns = 4, .min_column_width = 160,
            .gap = 12) {
      cui_metric(ui, "Accuracy", accuracy, .icon = CUI_ICON_TARGET);
      cui_metric(ui, "Correct rows", correct, .icon = CUI_ICON_BADGE_CHECK);
      cui_metric(ui, "Rows checked", rows, .icon = CUI_ICON_LIST);
      cui_metric(ui, "Skipped", skipped, .icon = CUI_ICON_SKIP_FORWARD);
    }
  } else if (result->scored) {
    char mse[48];
    char rmse[48];
    char mae[48];
    snprintf(mse, sizeof mse, "%.6f", result->mse);
    snprintf(rmse, sizeof rmse, "%.6f", sqrt(result->mse));
    snprintf(mae, sizeof mae, "%.6f", result->mae);
    cui_box(ui, .dir = CUI_GRID, .columns = 4, .min_column_width = 160,
            .gap = 12) {
      cui_metric(ui, "MSE", mse, .icon = CUI_ICON_ACTIVITY);
      cui_metric(ui, "RMSE", rmse, .icon = CUI_ICON_TRENDING_DOWN);
      cui_metric(ui, "MAE", mae, .icon = CUI_ICON_ACTIVITY);
      cui_metric(ui, "Rows checked", rows, .icon = CUI_ICON_LIST);
    }
  } else {
    cui_box(ui, .dir = CUI_GRID, .columns = 3, .min_column_width = 160,
            .gap = 12) {
      cui_metric(ui, "Rows predicted", rows, .icon = CUI_ICON_LIST);
      cui_metric(ui, "Scoring", "Off", .icon = CUI_ICON_EYE_OFF);
      if (result->skipped > 0u) {
        cui_metric(ui, "Skipped", skipped, .icon = CUI_ICON_SKIP_FORWARD);
      }
    }
  }
}

static void studio_eval_column_names(const elml_eval_result *result,
                                     const uint32_t *columns, uint32_t count,
                                     char *out, size_t cap)
{
  if (out == nullptr || cap == 0u) {
    return;
  }
  if (columns == nullptr || count == 0u) {
    snprintf(out, cap, "%s", "-");
    return;
  }

  out[0] = '\0';
  size_t used = 0u;
  for (uint32_t i = 0u; i < count; ++i) {
    char fallback[32];
    const char *name = nullptr;
    if (result->has_header) {
      name = elml_csv_cell(&result->csv, 0u, columns[i]);
    }
    if (name == nullptr || name[0] == '\0') {
      snprintf(fallback, sizeof fallback, "Column %u", columns[i] + 1u);
      name = fallback;
    }
    if (used + 2u >= cap) {
      break;
    }
    const int written = snprintf(out + used, cap - used, "%s%.200s",
                                 (i == 0u) ? "" : ", ", name);
    if (written < 0 || (size_t)written >= cap - used) {
      out[cap - 1u] = '\0';
      return;
    }
    used += (size_t)written;
  }
}

static void studio_eval_model_card(cui_ctx *ui, const studio_state *state)
{
  const elml_eval_result *result = &state->eval_result;

  cui_card(ui, .width = CUI_FILL, .gap = 10) {
    cui_row(ui, .width = CUI_FILL, .align = CUI_ALIGN_CENTER) {
      cui_text(ui, "Model and columns", .font = CUI_FONT_MEDIUM, .size = 15);
      cui_spacer(ui);
      if (elml_eval_is_classification(&result->model)) {
        cui_badge(ui, "Classification", .tone = CUI_TONE_VIOLET, .dot = true);
      } else {
        cui_badge(ui, "Regression", .tone = CUI_TONE_TEAL, .dot = true);
      }
    }

    studio_detail(ui, "Model", state->eval_model_path);
    char shape[96];
    snprintf(shape, sizeof shape, "%u input%s -> %u output%s",
             result->model.input.dim[0],
             (result->model.input.dim[0] == 1u) ? "" : "s", result->output_dim,
             (result->output_dim == 1u) ? "" : "s");
    studio_detail(ui, "Shape", shape);

    char input_names[ELML_CONFIG_LIST_MAX];
    char target_names[ELML_CONFIG_LIST_MAX];
    studio_eval_column_names(result, result->input_columns,
                             result->input_count, input_names,
                             sizeof input_names);
    studio_eval_column_names(result, result->target_columns,
                             result->target_count, target_names,
                             sizeof target_names);
    studio_detail(ui, "Inputs",
                  (result->input_count > 0u) ? input_names : "-");
    studio_detail(ui, "Targets",
                  (result->scored && result->target_count > 0u)
                      ? target_names
                      : "not scored");
    if (result->skipped > 0u && result->skipped_note[0] != '\0') {
      cui_muted(ui, result->skipped_note);
    }
  }
}

static void studio_eval_table(cui_ctx *ui, studio_state *state)
{
  const elml_eval_result *result = &state->eval_result;
  const bool errors_only =
      result->classification && result->scored && state->eval_show_errors_only;

  if (result->row_count == 0u) {
    cui_empty_state(ui, "No rows evaluated", .icon = CUI_ICON_INBOX,
                    .description = "The CSV has no data rows to check.");
    return;
  }

  uint32_t matching = 0u;
  for (uint32_t r = 0u; r < result->row_count; ++r) {
    if (!errors_only || !result->rows[r].correct) {
      matching++;
    }
  }

  const uint32_t limit = (state->eval_display_limit == 0u)
                             ? matching
                             : state->eval_display_limit;
  const uint32_t show = (matching > limit) ? limit : matching;

  cui_card(ui, .width = CUI_FILL, .gap = 12) {
    cui_row(ui, .width = CUI_FILL, .align = CUI_ALIGN_CENTER, .wrap = true) {
      cui_text(ui, "Per-row results", .font = CUI_FONT_MEDIUM, .size = 15);
      cui_spacer(ui);
      if (result->classification && result->scored) {
        (void)cui_switch(ui, "Incorrect only",
                         &state->eval_show_errors_only);
        cui_tooltip(ui, "Hide rows the model classified correctly.");
      }
    }

    if (matching == 0u) {
      cui_empty_state(ui, "Every checked row is correct",
                      .icon = CUI_ICON_CHECK_CHECK,
                      .description = "Turn off the filter to see all rows.");
    } else if (result->classification) {
      cui_table_column columns[] = {
          {"Line", CUI_PX(72), CUI_TEXT_RIGHT, false},
          {"Target", CUI_FILL, CUI_TEXT_LEFT, false},
          {"Prediction", CUI_FILL, CUI_TEXT_LEFT, false},
          {"Confidence", CUI_PX(110), CUI_TEXT_RIGHT, false},
          {"Result", CUI_PX(118), CUI_TEXT_LEFT, false},
      };
      uint32_t displayed = 0u;
      cui_table(ui, "eval_rows", columns, 5, .height = CUI_PX(460),
                .striped = true, .compact = true) {
        for (uint32_t r = 0u; r < result->row_count && displayed < show; ++r) {
          const elml_eval_row *row = &result->rows[r];
          if (errors_only && row->correct) {
            continue;
          }
          displayed++;

          (void)cui_table_row(ui, false);
          cui_table_cell_textf(ui, "%u", row->line);

          const char *target = nullptr;
          if (result->scored && result->target_columns != nullptr) {
            target = elml_csv_cell(&result->csv, row->line - 1u,
                                   result->target_columns[0]);
          }
          cui_table_cell_text(ui, (target != nullptr) ? target : "-");

          char label[128];
          cui_table_cell_text(
              ui, elml_eval_class_label(&result->model, row->predicted, label,
                                        sizeof label));

          char confidence[32];
          snprintf(confidence, sizeof confidence, "%.1f%%",
                   (double)row->confidence * 100.0);
          cui_table_cell_text(ui, confidence);

          cui_table_cell(ui);
          if (!result->scored) {
            cui_text(ui, "-", .color = CUI_COLOR_TEXT_SECONDARY);
          } else if (row->expected < 0) {
            cui_badge(ui, "No target", .tone = CUI_TONE_NEUTRAL);
          } else if (row->correct) {
            cui_badge(ui, "Correct", .tone = CUI_TONE_SUCCESS, .dot = true);
          } else {
            cui_badge(ui, "Wrong", .tone = CUI_TONE_DANGER, .dot = true);
          }
        }
      }
    } else {
      cui_table_column columns[] = {
          {"Line", CUI_PX(72), CUI_TEXT_RIGHT, false},
          {"Target", CUI_FILL, CUI_TEXT_LEFT, false},
          {"Prediction", CUI_FILL, CUI_TEXT_LEFT, false},
          {"Abs error", CUI_PX(150), CUI_TEXT_RIGHT, false},
      };
      uint32_t displayed = 0u;
      cui_table(ui, "eval_rows_regression", columns, 4, .height = CUI_PX(460),
                .striped = true, .compact = true) {
        for (uint32_t r = 0u; r < result->row_count && displayed < show; ++r) {
          const elml_eval_row *row = &result->rows[r];
          displayed++;

          (void)cui_table_row(ui, false);
          cui_table_cell_textf(ui, "%u", row->line);

          char target_text[512];
          char prediction_text[512];
          char error_text[512];
          studio_eval_vector_text(elml_eval_targets(result, r),
                                  result->output_dim, target_text,
                                  sizeof target_text);
          studio_eval_vector_text(elml_eval_values(result, r),
                                  result->output_dim, prediction_text,
                                  sizeof prediction_text);
          studio_eval_error_text(result, row, error_text, sizeof error_text);

          cui_table_cell_text(ui, target_text);
          cui_table_cell_text(ui, prediction_text);
          cui_table_cell_text(ui, error_text);
        }
      }
    }

    if (matching > show) {
      char note[160];
      snprintf(note, sizeof note,
               "Showing the first %u of %u matching rows; save the predictions "
               "CSV for the complete list.",
               show, matching);
      cui_muted(ui, note);
    } else if (errors_only) {
      char note[128];
      snprintf(note, sizeof note, "%u incorrect row%s shown.", matching,
               (matching == 1u) ? "" : "s");
      cui_muted(ui, note);
    }
  }
}

static bool studio_eval_view(cui_ctx *ui, studio_state *state)
{
  const bool load_config = studio_eval_controls(ui, state);

  if (state->eval_has_result && state->eval_ok) {
    studio_eval_metrics(ui, state);
    studio_eval_model_card(ui, state);
    studio_eval_table(ui, state);
    if (state->eval_scroll_results) {
      state->eval_scroll_results = false;
      cui_scroll_to(ui, "studio_eval_content", 620.0f);
    }
  }

  return load_config;
}

/* ------------------------------------------------------------------------- */
/* Main panel                                                                */
/* ------------------------------------------------------------------------- */

static void studio_training_view(cui_ctx *ui, studio_state *state,
                               const studio_view *view)
{
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
    cui_box(ui, .dir = CUI_GRID, .columns = 2, .min_column_width = 280,
            .gap = 14) {
      studio_data_split(ui, view);
      studio_run_log(ui, view);
    }
    studio_run_details(ui, state, view);
  }
}

static void studio_main(cui_ctx *ui, studio_state *state,
                        const studio_view *view)
{
  static const char *tabs[] = {"Training", "Evaluate"};
  static const char *tab_icons[] = {CUI_ICON_ACTIVITY, CUI_ICON_TARGET};
  const bool evaluating = studio_eval_running(state);
  bool load_config = false;

  cui_column(ui, .width = CUI_FILL, .height = CUI_FILL,
             .bg = CUI_COLOR_BACKGROUND) {
    cui_row(ui, .width = CUI_FILL, .pad_x = 24, .pad_y = 14, .gap = 12,
            .align = CUI_ALIGN_CENTER) {
      cui_column(ui, .gap = 2) {
        cui_heading(ui, (state->tab == 0) ? "Training run"
                                         : "Evaluate a saved model",
                    2);
        cui_secondary(ui,
                      (state->tab == 0)
                          ? "Adjust parameters on the left; the charts follow "
                            "each epoch."
                          : "Load a CSV and a model, then check every row.");
      }
      cui_spacer(ui);
      if (state->tab == 0) {
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
      } else {
        if (evaluating) {
          cui_badge(ui, "Evaluating", .tone = CUI_TONE_INFO, .dot = true);
        } else if (state->eval_error[0] != '\0') {
          cui_badge(ui, "Needs attention", .tone = CUI_TONE_DANGER, .dot = true);
        } else if (state->eval_has_result) {
          cui_badge(ui, "Complete", .tone = CUI_TONE_SUCCESS, .dot = true);
        } else {
          cui_badge(ui, "Idle", .tone = CUI_TONE_NEUTRAL, .dot = true);
        }
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

    cui_row(ui, .width = CUI_FILL, .pad_x = 24, .gap = 4) {
      cui_tabs(ui, "studio_tabs", tabs, 2, &state->tab, .icons = tab_icons);
    }
    cui_separator(ui);

    if (state->tab == 0) {
      studio_training_view(ui, state, view);
    } else {
      cui_scroll(ui, .id = "studio_eval_content", .width = CUI_FILL,
                 .height = CUI_FILL, .pad = 24, .gap = 18) {
        load_config = studio_eval_view(ui, state);
      }
    }
  }

  if (load_config) {
    studio_load_config(state, false);
    studio_lock(state);
    if (state->snap.error[0] != '\0') {
      snprintf(state->eval_error, sizeof state->eval_error, "%.1000s",
               state->snap.error);
      snprintf(state->eval_status, sizeof state->eval_status, "%s",
               "The configuration could not be loaded.");
    }
    studio_unlock(state);
    cui_request_redraw(ui);
  }
}

static void studio_frame(cui_ctx *ui, studio_state *state)
{
  studio_eval_collect(state);
  studio_columns_sync(state);
  studio_eval_columns_sync(state);

  studio_view view;
  studio_take_view(state, &view);

  const bool narrow = cui_viewport_size(ui).x < 980.0f;
  bool load_clicked = false;

  cui_row(ui, .width = CUI_FILL, .height = CUI_FILL, .gap = 0,
          .align = CUI_ALIGN_STRETCH, .bg = CUI_COLOR_BACKGROUND) {
    if (state->tab == 0) {
      load_clicked = studio_sidebar(ui, state, view.running);
      if (!narrow) {
        cui_separator(ui);
      }
    }
    studio_main(ui, state, &view);
  }

  if (load_clicked) {
    studio_load_config(state, false);
  }

  /* Idle windows sleep inside CUI. While a run is active, ask for the next
   * frame explicitly so the worker's epoch snapshots reach the charts. */
  if (view.running || studio_eval_running(state)) {
    cui_request_redraw(ui);
  }
}

#if !defined(ELML_WASM)
static bool studio_parse_args(studio_state *state, int argc, char **argv,
                              bool *autostart, bool *autoeval)
{
  bool explicit_config = false;
  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "--autostart") == 0) {
      *autostart = true;
    } else if (strcmp(argv[i], "--autoeval") == 0) {
      *autoeval = true;
    } else if (strcmp(argv[i], "--evaluate") == 0) {
      state->tab = 1;
    } else if (strcmp(argv[i], "--config") == 0 && i + 1 < argc) {
      studio_copy_string(state->config_path, sizeof state->config_path,
                         argv[i + 1]);
      explicit_config = true;
      ++i;
    } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
      printf("usage: elml_studio [--config FILE] [--evaluate] [--autostart] "
             "[--autoeval]\n"
             "\n"
             "  --config FILE   load an .ini training configuration\n"
             "  --evaluate      open the Evaluate tab\n"
             "  --autostart     start training as soon as the window opens\n"
             "  --autoeval      evaluate the configured CSV and model on\n"
             "                  startup (implies --evaluate)\n"
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
  if (*autoeval) {
    state->tab = 1;
  }
  if (state->config_path[0] != '\0') {
    studio_load_config(state, !explicit_config);
  }
  return true;
}
#endif /* !ELML_WASM */

#if defined(ELML_WASM)
/* ------------------------------------------------------------------------- */
/* Browser glue                                                              */
/* ------------------------------------------------------------------------- */

static studio_state *studio_web_current;

/* Prototypes for the exports below, so -Wmissing-prototypes stays quiet. */
int elml_studio_set_file(int kind, const char *name, const void *data,
                         int len);
void elml_studio_start(void);
void elml_studio_stop(void);
void elml_studio_shutdown(void);
void elml_studio_evaluate(void);
int elml_studio_epoch(void);
int elml_studio_running(void);
int elml_studio_eval_running(void);
int elml_studio_finished(void);
int elml_studio_exported(void);
const char *elml_studio_status(void);
const char *elml_studio_error(void);
float elml_studio_loss(void);
float elml_studio_accuracy(void);
int elml_studio_has_accuracy(void);
int elml_studio_epochs(void);
int elml_studio_best_epoch(void);
int elml_studio_train_rows(void);
int elml_studio_val_rows(void);
const char *elml_studio_model(void);
const char *elml_studio_algorithm(void);
int elml_studio_tab(void);
void elml_studio_set_tab(int tab);
const char *elml_studio_eval_status(void);
const char *elml_studio_eval_error(void);
int elml_studio_eval_has_result(void);
int elml_studio_eval_correct(void);
int elml_studio_file_count(void);
const char *elml_studio_file_name(int index);
int elml_studio_file_size(int index);
const uint8_t *elml_studio_read_file(const char *path, int *len);

static void studio_web_frame(cui_ctx *ui, void *user)
{
  studio_state *state = (studio_state *)user;
  state->ui = ui;
  studio_web_pump(state);
  studio_frame(ui, state);
}

/* kind: 0 config, 1 training CSV, 2 validation CSV, 3 model, 4 evaluation CSV.
 * The page registers the bytes first, then calls this. */
STUDIO_EXPORT(elml_studio_set_file)
int elml_studio_set_file(int kind, const char *name, const void *data, int len)
{
  studio_state *state = studio_web_current;
  if (state == nullptr || name == nullptr || len < 0) {
    return 1;
  }
  if (!elml_memfs_put(name, data, (size_t)len)) {
    return 2;
  }

  switch (kind) {
  case 0:
    snprintf(state->config_path, sizeof state->config_path, "%s", name);
    studio_load_config(state, false);
    break;
  case 1:
    snprintf(state->csv_path, sizeof state->csv_path, "%s", name);
    (void)studio_columns_load(state, true);
    break;
  case 2:
    snprintf(state->val_csv_path, sizeof state->val_csv_path, "%s", name);
    break;
  case 3:
    snprintf(state->eval_model_path, sizeof state->eval_model_path, "%s",
             name);
    if (state->eval_has_result) {
      elml_eval_free(&state->eval_result);
      state->eval_has_result = false;
      state->eval_ok = false;
      state->eval_error[0] = '\0';
      snprintf(state->eval_status, sizeof state->eval_status, "%s",
               "Model changed; evaluate again to refresh the results.");
    }
    break;
  case 4:
    snprintf(state->eval_csv_path, sizeof state->eval_csv_path, "%s", name);
    (void)studio_eval_columns_load(state, true);
    if (state->eval_has_result) {
      elml_eval_free(&state->eval_result);
      state->eval_has_result = false;
      state->eval_ok = false;
      state->eval_error[0] = '\0';
      snprintf(state->eval_status, sizeof state->eval_status, "%s",
               "CSV changed; evaluate again to refresh the results.");
    }
    break;
  default:
    return 3;
  }

  if (studio_ui(state) != nullptr) {
    cui_request_redraw(studio_ui(state));
  }
  return 0;
}

STUDIO_EXPORT(elml_studio_start)
void elml_studio_start(void)
{
  if (studio_web_current != nullptr) {
    studio_start(studio_web_current);
  }
}

STUDIO_EXPORT(elml_studio_stop)
void elml_studio_stop(void)
{
  if (studio_web_current != nullptr) {
    studio_stop(studio_web_current);
  }
}

/* Called from the page's pagehide handler: drop the training session and the
 * evaluation results before the module goes away. */
STUDIO_EXPORT(elml_studio_shutdown)
void elml_studio_shutdown(void)
{
  if (studio_web_current != nullptr) {
    studio_shutdown(studio_web_current);
  }
}

STUDIO_EXPORT(elml_studio_evaluate)
void elml_studio_evaluate(void)
{
  if (studio_web_current != nullptr) {
    studio_eval_start(studio_web_current);
  }
}

/* Small read-only accessors for the page and the Node smoke test. */
STUDIO_EXPORT(elml_studio_epoch)
int elml_studio_epoch(void)
{
  return (studio_web_current != nullptr)
             ? (int)studio_web_current->snap.epoch
             : 0;
}

STUDIO_EXPORT(elml_studio_running)
int elml_studio_running(void)
{
  return (studio_web_current != nullptr && studio_web_current->running) ? 1 : 0;
}

STUDIO_EXPORT(elml_studio_eval_running)
int elml_studio_eval_running(void)
{
  return (studio_web_current != nullptr && studio_web_current->eval_running) ? 1
                                                                            : 0;
}

STUDIO_EXPORT(elml_studio_finished)
int elml_studio_finished(void)
{
  return (studio_web_current != nullptr && studio_web_current->finished) ? 1 : 0;
}

STUDIO_EXPORT(elml_studio_exported)
int elml_studio_exported(void)
{
  return (studio_web_current != nullptr && studio_web_current->snap.exported)
             ? 1
             : 0;
}

STUDIO_EXPORT(elml_studio_status)
const char *elml_studio_status(void)
{
  return (studio_web_current != nullptr) ? studio_web_current->snap.status : "";
}

STUDIO_EXPORT(elml_studio_error)
const char *elml_studio_error(void)
{
  return (studio_web_current != nullptr) ? studio_web_current->snap.error : "";
}

STUDIO_EXPORT(elml_studio_loss)
float elml_studio_loss(void)
{
  return (studio_web_current != nullptr) ? studio_web_current->snap.loss : 0.0f;
}

STUDIO_EXPORT(elml_studio_accuracy)
float elml_studio_accuracy(void)
{
  return (studio_web_current != nullptr) ? studio_web_current->snap.accuracy
                                         : 0.0f;
}

STUDIO_EXPORT(elml_studio_has_accuracy)
int elml_studio_has_accuracy(void)
{
  return (studio_web_current != nullptr && studio_web_current->snap.has_accuracy)
             ? 1
             : 0;
}

STUDIO_EXPORT(elml_studio_epochs)
int elml_studio_epochs(void)
{
  return (studio_web_current != nullptr) ? (int)studio_web_current->snap.epochs
                                         : 0;
}

STUDIO_EXPORT(elml_studio_best_epoch)
int elml_studio_best_epoch(void)
{
  return (studio_web_current != nullptr)
             ? (int)studio_web_current->snap.best_epoch
             : 0;
}

STUDIO_EXPORT(elml_studio_train_rows)
int elml_studio_train_rows(void)
{
  return (studio_web_current != nullptr)
             ? (int)studio_web_current->snap.train_rows
             : 0;
}

STUDIO_EXPORT(elml_studio_val_rows)
int elml_studio_val_rows(void)
{
  return (studio_web_current != nullptr) ? (int)studio_web_current->snap.val_rows
                                         : 0;
}

STUDIO_EXPORT(elml_studio_model)
const char *elml_studio_model(void)
{
  return (studio_web_current != nullptr) ? studio_web_current->snap.model : "";
}

STUDIO_EXPORT(elml_studio_algorithm)
const char *elml_studio_algorithm(void)
{
  return (studio_web_current != nullptr) ? studio_web_current->snap.algorithm
                                         : "";
}

STUDIO_EXPORT(elml_studio_tab)
int elml_studio_tab(void)
{
  return (studio_web_current != nullptr) ? studio_web_current->tab : 0;
}

STUDIO_EXPORT(elml_studio_set_tab)
void elml_studio_set_tab(int tab)
{
  if (studio_web_current != nullptr) {
    studio_web_current->tab = (tab == 1) ? 1 : 0;
    if (studio_ui(studio_web_current) != nullptr) {
      cui_request_redraw(studio_ui(studio_web_current));
    }
  }
}

STUDIO_EXPORT(elml_studio_eval_status)
const char *elml_studio_eval_status(void)
{
  return (studio_web_current != nullptr) ? studio_web_current->eval_status
                                         : "";
}

STUDIO_EXPORT(elml_studio_eval_error)
const char *elml_studio_eval_error(void)
{
  return (studio_web_current != nullptr) ? studio_web_current->eval_error : "";
}

STUDIO_EXPORT(elml_studio_eval_has_result)
int elml_studio_eval_has_result(void)
{
  return (studio_web_current != nullptr &&
          studio_web_current->eval_has_result)
             ? 1
             : 0;
}

STUDIO_EXPORT(elml_studio_eval_correct)
int elml_studio_eval_correct(void)
{
  return (studio_web_current != nullptr)
             ? (int)studio_web_current->eval_result.correct
             : 0;
}

STUDIO_EXPORT(elml_studio_file_count)
int elml_studio_file_count(void)
{
  return elml_memfs_count();
}

STUDIO_EXPORT(elml_studio_file_name)
const char *elml_studio_file_name(int index)
{
  return elml_memfs_name(index);
}

STUDIO_EXPORT(elml_studio_file_size)
int elml_studio_file_size(int index)
{
  const size_t size = elml_memfs_size(index);
  return (size > (size_t)INT_MAX) ? INT_MAX : (int)size;
}

STUDIO_EXPORT(elml_studio_read_file)
const uint8_t *elml_studio_read_file(const char *path, int *len)
{
  const uint8_t *data = nullptr;
  size_t size = 0u;
  if (!elml_memfs_get(path, &data, &size)) {
    return nullptr;
  }
  if (len != nullptr) {
    *len = (size > (size_t)INT_MAX) ? INT_MAX : (int)size;
  }
  return data;
}
#endif /* ELML_WASM */

#if defined(ELML_WASM)
int main(void)
#else
int main(int argc, char **argv)
#endif
{
  studio_state *state = (studio_state *)calloc(1u, sizeof *state);
  if (state == nullptr) {
    fprintf(stderr, "elml_studio: out of memory\n");
    return 1;
  }
#if !defined(ELML_WASM)
  if (pthread_mutex_init(&state->mutex, nullptr) != 0) {
    fprintf(stderr, "elml_studio: cannot create the UI mutex\n");
    free(state);
    return 1;
  }
#endif

  studio_defaults(state);

#if defined(ELML_WASM)
  /*
   * The page's loader calls web_boot -> web_app_main (this function). CUI
   * records the frame callback and returns; web/host drives it afterwards.
   * The state therefore outlives this call and is never freed.
   */
  studio_web_current = state;
  (void)cui_app_run(&(cui_app_desc){
                        .title = "embedlml studio",
                        .width = 1320,
                        .height = 880,
                        .theme = CUI_THEME_DARK,
                        .accent = CUI_ACCENT_VIOLET},
                    studio_web_frame, state);
  return 0;
#else
  bool autostart = false;
  bool autoeval = false;
  if (!studio_parse_args(state, argc, argv, &autostart, &autoeval)) {
    pthread_mutex_destroy(&state->mutex);
    free(state);
    return 0;
  }

  cui_app *app = cui_app_create(&(cui_app_desc){
      .title = "embedlml studio",
      .width = 1320,
      .height = 880,
      .min_width = 860,
      .min_height = 600,
      .continuous = false,
      .theme = CUI_THEME_DARK,
      .accent = CUI_ACCENT_VIOLET});
  if (app == nullptr) {
    fprintf(stderr, "elml_studio: cannot create a window\n");
    pthread_mutex_destroy(&state->mutex);
    free(state);
    return 1;
  }
  state->app = app;
  state->ui = cui_app_ui(app);

  bool start_requested = autostart;
  bool eval_requested = autoeval;
  const bool screenshot_mode = getenv("CUI_SCREENSHOT") != nullptr;
  double active_until = 0.0;
  while (cui_app_frame(app)) {
    studio_frame(state->ui, state);
    if (start_requested) {
      start_requested = false;
      studio_start(state);
    }
    if (eval_requested) {
      eval_requested = false;
      studio_eval_start(state);
    }
    if (!screenshot_mode) {
      studio_pace(state->ui,
                  studio_is_running(state) || studio_eval_running(state),
                  &active_until);
    }
  }

  studio_shutdown(state);
  cui_app_destroy(app);
  studio_columns_release(&state->columns);
  studio_columns_release(&state->eval_columns);
  pthread_mutex_destroy(&state->mutex);
  free(state);
  return 0;
#endif
}
