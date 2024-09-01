import os
import signal as os_signal

import statsmodels.api as sm
import librosa
import numpy as np
from pasimple import PaSimple, PA_STREAM_RECORD, PA_SAMPLE_S16LE, PA_STREAM_PLAYBACK, exceptions as pa_exceptions
from scipy import signal

PLAY_DEVICE_NAME = os.getenv('PLAY_DEVICE_NAME', None)
PULSE_SERVER = os.getenv('PULSE_SERVER', None)
RECORD_DEVICE_NAMES = os.getenv('RECORD_DEVICE_NAMES', None).split(',')
CHUNK_SIZE = int(os.getenv('RECORD_CHUNK_SIZE', '2048'))
MAX_SIZE = int(os.getenv('MAX_SIZE', '65536'))
RECORD_RATE = int(os.getenv('RECORD_RATE', '16000'))
CORRELATION_THRESHOLD = float(os.getenv('CORRELATION_THRESHOLD', '0.01'))
lag_limit_pulses = round(float(os.getenv('LAG_LIMIT_MS', '750')) * RECORD_RATE / 1000)
DECAY_HALF_LIFE_SEC = float(os.getenv('DECAY_HALF_LIFE_SEC', '30'))
PLAY_BUFFER_SIZE = int(os.getenv('play_buffer_size', CHUNK_SIZE))
decay_coefficient = 2 ** (-1 / (DECAY_HALF_LIFE_SEC * RECORD_RATE / MAX_SIZE / 2))  # 2 times per MAX_SIZE a lag decays
add_points = 3  # x 2 because of positive and negative correlation
max_play_buffer_diff = round(PLAY_BUFFER_SIZE * 0.01)

assert len(RECORD_DEVICE_NAMES) >= 2
assert MAX_SIZE >= 2 * lag_limit_pulses
assert CHUNK_SIZE >= 512
assert MAX_SIZE >= 2 * CHUNK_SIZE

record_pas = []
pa_play = None
record = True
play_buffer_filling = True


def pa_init():
    for device in RECORD_DEVICE_NAMES:
        try:
            record_pas.append(PaSimple(
                PA_STREAM_RECORD,
                PA_SAMPLE_S16LE,
                1,
                RECORD_RATE,
                'voice-kodi merger recorder',
                None,
                PULSE_SERVER,
                device,
                maxlength=2*CHUNK_SIZE*2,
                fragsize=CHUNK_SIZE*2,
            ))
        except pa_exceptions.PaSimpleError as exc:
            print(f"Connect to record device <<{device}>> failure! {exc}")
            raise exc

    global pa_play
    pa_play = PaSimple(
        PA_STREAM_PLAYBACK,
        PA_SAMPLE_S16LE,
        1,
        RECORD_RATE,
        'voice-kodi merger',
        None,
        PULSE_SERVER,
        PLAY_DEVICE_NAME,
        maxlength=4 * CHUNK_SIZE * 2,
        prebuf=2*CHUNK_SIZE * 2,
    )
    global record
    global play_buffer_filling
    record = True
    play_buffer_filling = True


def signal_handler(sig, frame):
    global record
    record = False
    save_state()


os_signal.signal(os_signal.SIGINT, signal_handler)
os_signal.signal(os_signal.SIGTERM, signal_handler)

sort_lag_array_max = np.array([-abs(x) for x in range(-lag_limit_pulses, lag_limit_pulses)])
sort_lag_array_min = np.array([abs(x) for x in range(-lag_limit_pulses, lag_limit_pulses)])


def what_shift_print(correlations, only_fist_n, prefix=''):
    top_lags_max = np.lexsort((sort_lag_array_max, correlations))
    top_lags_max = np.flip(top_lags_max[-only_fist_n:])
    top_lags_min = np.lexsort((sort_lag_array_min, correlations))
    top_lags_min = top_lags_min[:only_fist_n]
    sorted_stored_lag_indexes = np.concat((top_lags_max, top_lags_min))
    # print(f'{prefix}correlation lags:\t\t', top_lags_max - lag_limit_pulses, top_lags_min - lag_limit_pulses, )
    print('Corresponding correlation values:\t', ' '.join('{0:.3f}'.format(round(x, 4)) for x in correlations[sorted_stored_lag_indexes]))
    max_lag_index = top_lags_max[0]
    min_lag_index = top_lags_min[0]
    straight = correlations[max_lag_index] >= - correlations[min_lag_index]
    if straight:
        shift = max_lag_index
    else:
        shift = min_lag_index
    shift -= lag_limit_pulses
    print(f'{prefix}selected shift:\t', shift, '\t', straight)

    return shift, straight, top_lags_max, top_lags_min


def save_state():
    np.savez(
        f'cache/{"_plus_".join(RECORD_DEVICE_NAMES)}_{RECORD_RATE}_{lag_limit_pulses}.npz',
        desired_shifts=desired_shifts,
        sounds_lags=sounds_lags,
        clock_screws=clock_screws,
    )


def load_state():
    try:
        data = np.load(f'cache/{"_plus_".join(RECORD_DEVICE_NAMES)}_{RECORD_RATE}_{lag_limit_pulses}.npz')
        global desired_shifts
        global sounds_lags
        global clock_screws
        desired_shifts = data['desired_shifts']
        sounds_lags = data['sounds_lags']
        clock_screws = data['clock_screws']
        print('Loaded state successfully. Clock screw:', clock_screws)
    except FileNotFoundError:
        print("No saved state found. Initializing default values.")
        # Assuming record_pas and lag_limit_pulses are defined earlier in your program


pa_init()
to_play = np.zeros((PLAY_BUFFER_SIZE), dtype=np.float64)

sound_lengths = [0 for x in record_pas]
sounds = [np.zeros((MAX_SIZE), np.float64) for x in record_pas]
history_len = 64
desired_shifts_history_len = 0
desired_shifts_history = np.zeros((history_len, len(record_pas)), dtype=np.int32)
desired_shifts_history_weights = np.zeros((history_len, len(record_pas)))
desired_shifts = np.zeros((len(record_pas)), dtype=np.int32)
actual_playeds = np.zeros((len(record_pas)), dtype=np.int64)
sounds_lags = np.zeros((len(record_pas) - 1, 2 * lag_limit_pulses), dtype=np.float64)
minus = np.zeros(len(record_pas), np.int64)
recorded_index = 0
clock_screws = np.zeros((len(record_pas)), dtype=np.float64)
load_state()


def play_merged():
    global to_play
    # normalize shifts - doesn't matter lags - matter lags difference - we can move them
    past_actual_sorted_indexes = np.argsort(actual_playeds)
    min_actual_shift = actual_playeds[past_actual_sorted_indexes[0]] - minus[past_actual_sorted_indexes[0]]
    desired_shift_max = np.max(desired_shifts)
    desired_shift_min = np.min(desired_shifts)
    curr_actual_shift_diff = np.max(actual_playeds) - np.min(actual_playeds)
    # print(desired_shifts, sep=' ')
    # try to shift everything to use less overall lag:
    desired_sh = (desired_shifts + -desired_shift_max + np.min(sound_lengths) - PLAY_BUFFER_SIZE).astype(np.int64)
    # print(desired_sh, actual_playeds - minus, end=' ')

    desired_shift_min = np.min(desired_sh)
    # print(' : ', desired_sh - desired_shift_min, actual_playeds - minus - min_actual_shift, end='\t')
    # print('\t\t\t\t\t\t\t\t', end='')

    to_play.fill(0)
    for i, desired_shift in enumerate(desired_sh):
        actual_shift = actual_playeds[i] - minus[i]
        if sound_lengths[i] == MAX_SIZE:
            play_start_from = actual_shift
        else:
            play_start_from = sound_lengths[i] - PLAY_BUFFER_SIZE
        play_length_diff = 0
        # todo: we can grow buffer only if we can - add checking for size
        if actual_shift > desired_shift:
            # read from `sounds[i]` less than play_buffer_size => increase lag and use "more" "buffer"
            play_length_diff = min(max_play_buffer_diff, actual_shift - desired_shift)
            play_length_diff = -int(play_length_diff)

        elif actual_shift < desired_shift:
            # read from `sounds[i]` more than play_buffer_size
            play_length_diff = min(max_play_buffer_diff, desired_shift - actual_shift)
            play_length_diff = int(play_length_diff)
        assert play_start_from >= 0
        if play_length_diff:
            # print('\t', i, play_length_diff, play_start_from, end='')
            play_length = PLAY_BUFFER_SIZE + play_length_diff
            actual_playeds[i] += play_length
            to_play_tmp = sounds[i][play_start_from:play_start_from + play_length]
            to_play_tmp = to_play_tmp.astype(np.float32)  # / 32768.0
            stretched_audio = librosa.core.resample(to_play_tmp, orig_sr=play_length, target_sr=PLAY_BUFFER_SIZE)
            to_play += (stretched_audio / len(sounds))[:PLAY_BUFFER_SIZE]
        else:
            # print('\t', i, 0, play_start_from, end='')
            actual_playeds[i] += PLAY_BUFFER_SIZE
            to_play_tmp = sounds[i][play_start_from:play_start_from + PLAY_BUFFER_SIZE]
            to_play += to_play_tmp / len(sounds)  # / 32768.0
    # Convert back to PCM (needed?)
    # print(f'play latency {pa_play.get_latency()}')
    pa_play.write((to_play * 32768.0).astype(np.int16).tobytes())


def calc_clock_drift():
    results = []
    t1 = np.arange(history_len) * MAX_SIZE / 2 + MAX_SIZE / 2
    for i in range(1, desired_shifts_history.shape[1]):
        y = desired_shifts_history[:, i]
        X = sm.add_constant(t1)  # Create design matrix with constant for intercept
        weights = desired_shifts_history_weights[:, i]

        # Ensure no zero weights are passed to the regression model
        mask = weights > 0
        y_filtered = y[mask]
        X_filtered = X[mask]
        weights_filtered = weights[mask]
        if y_filtered.size < 4:
            continue
        # Create a weighted least squares model
        wls_model = sm.WLS(y_filtered, X_filtered, weights=weights_filtered)
        result = wls_model.fit()

        # Store the result
        results.append({
            'signal': i,
            'params': result.params,
            'pvalues': result.pvalues,
            'rsquared': result.rsquared
        })

        # Print the fitted parameters for each signal
        correlations_sum = weights.sum()
        print(
            f"Signal {i} - Intercept (lag): {result.params[0]}, Slope (k): {result.params[1]}, "
            f"R-squared: {result.rsquared}, P-values: {result.pvalues}, Prob (F-statistic): {result.f_pvalue}, "
            f"correlations_sum: {correlations_sum}\n",
            result.summary()
        )
        no_relations_p_value = 1 / (10 ** 6)
        if result.rsquared_adj > 0.95 and result.llf > -550 and result.f_pvalue < no_relations_p_value \
                and correlations_sum > history_len * CORRELATION_THRESHOLD * 4:
            print('Detected clock screw!')
            clock_screws[i] = round(clock_screws[i] * CHUNK_SIZE) / CHUNK_SIZE + result.params[1]


while record:
    for i, pulse_recorder in enumerate(record_pas):
        read_size_compensated = round((1 + clock_screws[i]) * CHUNK_SIZE)
        # print(f'read_size_compensated {read_size_compensated}  --  {CHUNK_SIZE} | latency {i} {pulse_recorder.get_latency()}')
        buffer_not_compensated = pulse_recorder.read(read_size_compensated * 2)  # 16bit - 2 bytes per pulse
        array_not_compensated = np.frombuffer(buffer_not_compensated, dtype='<i2')
        array_not_compensated = array_not_compensated.astype(np.float32) / 32768.0
        not_compensated_l = len(array_not_compensated)
        array = librosa.core.resample(array_not_compensated, orig_sr=read_size_compensated, target_sr=CHUNK_SIZE)
        l = len(array)

        cut = max(l + sound_lengths[i] - MAX_SIZE, 0)
        if cut > 0:
            minus[i] += cut
            sounds[i][0:sound_lengths[i] - cut] = sounds[i][cut:sound_lengths[i]]
            sounds[i][-l:] = array
            sound_lengths[i] = MAX_SIZE
        else:
            sounds[i][sound_lengths[i]:sound_lengths[i] + l] = array
            sound_lengths[i] += l
        if i == 0:
            recorded_index += len(array)
    # print(' times out of boundaries: ', minus)

    # print(' %', recorded_index % MAX_SIZE, recorded_index % MAX_SIZE % 4096)
    if play_buffer_filling:
        play_buffer_filling = sound_lengths[0] < PLAY_BUFFER_SIZE
    if (not play_buffer_filling) \
            and all(sound_lengths[i] - PLAY_BUFFER_SIZE >= actual_playeds[i] - minus[i] for i, arr in enumerate(sounds)):
        play_merged()

    if recorded_index % MAX_SIZE not in (0, MAX_SIZE >> 1) or recorded_index < MAX_SIZE:
        continue
    x = sounds[0]
    sqrt_x2 = np.sum(x ** 2) ** 0.5

    if sqrt_x2 == 0:
        print('-- x**2 == 0')
        desired_shifts_history[desired_shifts_history_len, :] = np.zeros(len(record_pas))
        desired_shifts_history_len += 1
        if desired_shifts_history_len == history_len:
            desired_shifts_history_len = 0
            print('\tdesired_shifts_history\n\n', desired_shifts_history, '\n', desired_shifts_history_weights, '\n\n')
            calc_clock_drift()
        continue

    lags = signal.correlation_lags(x.size, x.size, mode="same")  # x and y has the same size

    auto_x_correlation = signal.correlate(x, x, mode="same")
    auto_x_stored_correlation = auto_x_correlation / sqrt_x2
    auto_x_correlation = auto_x_stored_correlation / sqrt_x2
    # print('--')
    for i in range(1, len(sounds)):
        y = sounds[i][:sound_lengths[i]]
        desired_shifts_history_weights[desired_shifts_history_len, i] = 0
        if np.any(np.isnan(y)) or np.any(np.isinf(y)):
            print(f'y{i} contains NaN or Inf')
        sqrt_y2 = np.sum(y ** 2)
        if sqrt_y2 == 0:
            # print('-- y**2 == 0')
            continue
        sqrt_y2 = sqrt_y2 ** 0.5

        auto_y_correlation = signal.correlate(y, y, mode="same")
        auto_y_stored_correlation = auto_y_correlation / sqrt_y2
        auto_y_correlation = auto_y_stored_correlation / sqrt_y2

        correlation = signal.correlate(x, y, mode="same")
        stored_correlation = correlation / sqrt_y2
        correlation = stored_correlation / sqrt_x2

        # correlation -= (auto_x_correlation + auto_y_correlation) / 2
        # stored_correlation -= (auto_x_stored_correlation + auto_y_stored_correlation) / 2

        sorted_by_corr_lag_indexes = np.argsort(correlation)
        top_lag_indices = np.concat((
            np.flip(sorted_by_corr_lag_indexes)[:add_points],
            sorted_by_corr_lag_indexes[:add_points]
        ))
        # Extract the corresponding lags
        top_lags = lags[top_lag_indices]
        # Output the three most probable lags
        # print(RECORD_DEVICE_NAMES[i])
        # print("full top correlation lags:\t", top_lags)
        # print("Corresponding correlation values", correlation[top_lag_indices])

        valid_indices = np.where(np.logical_and(np.abs(lags) <= lag_limit_pulses, lags < lag_limit_pulses))[0]
        # Filtered lags and correlations within the specified range
        filtered_lags = lags[valid_indices]
        filtered_correlations = correlation[valid_indices]

        local_shift, _, _, _ = what_shift_print(filtered_correlations, add_points, 'Top\t\t')

        sounds_lag_i = i-1
        middle_index = len(correlation) >> 1
        filtered_correlation = correlation[middle_index - lag_limit_pulses:middle_index + lag_limit_pulses]
        sounds_lags[sounds_lag_i] = (sounds_lags[sounds_lag_i] * decay_coefficient
                                     + filtered_correlation * (1 - decay_coefficient))

        overall_shift, straight, top_lags_max, top_lags_min = what_shift_print(sounds_lags[sounds_lag_i], 5, 'Stored\t')
        # experimental hide auto
        if overall_shift == local_shift:
            l = middle_index - lag_limit_pulses - overall_shift
            r = middle_index + lag_limit_pulses - overall_shift
            l_overflow = max(0 - l, 0)
            r_overflow = max(r - len(auto_x_correlation), 0)
            l = l_overflow + l
            r = r - r_overflow
            sys_err_x_correlation = auto_x_correlation[l:r]
            sys_err_y_correlation = auto_y_correlation[l:r]
            illusions = (sys_err_x_correlation + sys_err_y_correlation) \
                        * (1 - decay_coefficient) * decay_coefficient \
                        * filtered_correlation[overall_shift + lag_limit_pulses] / 2
            sounds_lags[sounds_lag_i][l_overflow:2 * lag_limit_pulses - r_overflow] -= illusions
        if straight:
            correlation_value = sounds_lags[sounds_lag_i, top_lags_max[0]]
        else:
            correlation_value = sounds_lags[sounds_lag_i, top_lags_min[0]]
        if (straight and correlation_value < CORRELATION_THRESHOLD) \
                or (not straight and abs(correlation_value) < CORRELATION_THRESHOLD):
            # print('below threshold - skipp!')
            continue
        desired_shifts[i] = -overall_shift
        desired_shifts_history_weights[desired_shifts_history_len, i] = abs(correlation_value)
    desired_shifts_history_weights[desired_shifts_history_len, 0] = 1
    desired_shifts[0] = 0
    desired_shifts_history[desired_shifts_history_len,:] = desired_shifts
    desired_shifts_history_len += 1
    if desired_shifts_history_len == history_len:
        desired_shifts_history_len = 0
        print('\tdesired_shifts_history\n\n', desired_shifts_history, '\n', desired_shifts_history_weights, '\n\n')
        calc_clock_drift()

for pulse_recorder in record_pas:
    pulse_recorder.close()
