from typing import Tuple, List, Any

import numpy as np
import eng_to_ipa as ipa
from numpy import ndarray
from ortools.sat.python import cp_model
from dtwalign import dtw_from_distance_matrix
from string import punctuation


def avg_wer(wer_scores, combined_ref_len):
    return float(sum(wer_scores)) / float(combined_ref_len)


def _levenshtein_distance(ref, hyp):
    m = len(ref)
    n = len(hyp)

    # special case
    if ref == hyp:
        return 0
    if m == 0:
        return n
    if n == 0:
        return m
    # swapping to ensure that ref is always a longer text than hyp
    if m < n:
        ref, hyp = hyp, ref
        m, n = n, m

    # use O(min(m, n)) space
    distance = np.zeros((2, n + 1), dtype=np.int32)

    # initialize distance matrix
    for j in range(0, n + 1):
        distance[0][j] = j

    # calculate levenshtein distance
    for i in range(1, m + 1):
        prev_row_idx = (i - 1) % 2
        cur_row_idx = i % 2
        distance[cur_row_idx][0] = i
        for j in range(1, n + 1):
            if ref[i - 1] == hyp[j - 1]:
                distance[cur_row_idx][j] = distance[prev_row_idx][j - 1]
            else:
                s_num = distance[prev_row_idx][j - 1] + 1
                i_num = distance[cur_row_idx][j - 1] + 1
                d_num = distance[prev_row_idx][j] + 1
                distance[cur_row_idx][j] = min(s_num, i_num, d_num)

    return distance[m % 2][n]


def word_errors(reference, hypothesis, ignore_case=False, delimiter=' '):

    if ignore_case:
        reference = reference.lower()
        hypothesis = hypothesis.lower()

    ref_words = reference.split(delimiter)
    hyp_words = hypothesis.split(delimiter)

    edit_distance = _levenshtein_distance(ref_words, hyp_words)
    return float(edit_distance), len(ref_words)


def char_errors(reference, hypothesis, ignore_case=False, remove_space=False):

    if ignore_case:
        reference = reference.lower()
        hypothesis = hypothesis.lower()

    join_char = ' '
    if remove_space:
        join_char = ''

    reference = join_char.join(filter(None, reference.split(' ')))
    hypothesis = join_char.join(filter(None, hypothesis.split(' ')))

    edit_distance = _levenshtein_distance(reference, hypothesis)
    return float(edit_distance), len(reference)


def wer(reference, hypothesis, ignore_case=False, delimiter=' '):

    edit_distance, ref_len = word_errors(reference, hypothesis, ignore_case,
                                         delimiter)

    if ref_len == 0:
        raise ValueError("Reference's word number should be greater than 0.")

    wer = float(edit_distance) / ref_len
    return wer


def cer(reference, hypothesis, ignore_case=False, remove_space=False):
    edit_distance, ref_len = char_errors(reference, hypothesis, ignore_case,
                                         remove_space)

    if ref_len == 0:
        raise ValueError("Length of reference should be greater than 0.")

    cer = float(edit_distance) / ref_len
    return cer


#########################
#########################
offset_blank = 1
TIME_THRESHOLD_MAPPING = 5.0


def get_resulting_string(mapped_indices: np.array, words_estimated: list, words_real: list) \
        -> tuple[list[str | list[Any] | Any], list[int | Any]]:
    mapped_words = []
    mapped_words_indices = []
    WORD_NOT_FOUND_TOKEN = '-'
    number_of_real_words = len(words_real)
    for word_idx in range(number_of_real_words):
        position_of_real_word_indices = np.where(
            mapped_indices == word_idx)[0].astype(np.int64)

        if len(position_of_real_word_indices) == 0:
            mapped_words.append(WORD_NOT_FOUND_TOKEN)
            mapped_words_indices.append(-1)
            continue

        if len(position_of_real_word_indices) == 1:
            mapped_words.append(
                words_estimated[position_of_real_word_indices[0]])
            mapped_words_indices.append(position_of_real_word_indices[0])
            continue
        # Check which index gives the lowest error
        if len(position_of_real_word_indices) > 1:
            error = 99999
            best_possible_combination = ''
            best_possible_idx = -1
            for single_word_idx in position_of_real_word_indices:
                idx_above_word = single_word_idx >= len(words_estimated)
                if idx_above_word:
                    continue
                error_word = edit_distance_python(
                    words_estimated[single_word_idx], words_real[word_idx])
                if error_word < error:
                    error = error_word * 1
                    best_possible_combination = words_estimated[single_word_idx]
                    best_possible_idx = single_word_idx

            mapped_words.append(best_possible_combination)
            mapped_words_indices.append(best_possible_idx)
            continue

    return mapped_words, mapped_words_indices


def get_word_distance_matrix(words_estimated: list, words_real: list) -> np.array:
    number_of_real_words = len(words_real)
    number_of_estimated_words = len(words_estimated)

    word_distance_matrix = np.zeros(
        (number_of_estimated_words + offset_blank, number_of_real_words))
    for idx_estimated in range(number_of_estimated_words):
        for idx_real in range(number_of_real_words):
            word_distance_matrix[idx_estimated, idx_real] = edit_distance_python(
                words_estimated[idx_estimated], words_real[idx_real])

    if offset_blank == 1:
        for idx_real in range(number_of_real_words):
            word_distance_matrix[number_of_estimated_words, idx_real] = len(words_real[idx_real])
    return word_distance_matrix


def get_best_path_from_distance_matrix(word_distance_matrix):
    modelCpp = cp_model.CpModel()

    number_of_real_words = word_distance_matrix.shape[1]
    number_of_estimated_words = word_distance_matrix.shape[0] - 1

    number_words = np.maximum(number_of_real_words, number_of_estimated_words)

    estimated_words_order = [modelCpp.NewIntVar(0, int(
        number_words - 1 + offset_blank), 'w%i' % i) for i in range(number_words + offset_blank)]

    # They are in ascending order
    for word_idx in range(number_words - 1):
        modelCpp.Add(
            estimated_words_order[word_idx + 1] >= estimated_words_order[word_idx])

    total_phoneme_distance = 0
    real_word_at_time = {}
    for idx_estimated in range(number_of_estimated_words):
        for idx_real in range(number_of_real_words):
            real_word_at_time[idx_estimated, idx_real] = modelCpp.NewBoolVar(
                'real_word_at_time' + str(idx_real) + '-' + str(idx_estimated))
            modelCpp.Add(estimated_words_order[idx_estimated] == idx_real).OnlyEnforceIf(
                real_word_at_time[idx_estimated, idx_real])
            total_phoneme_distance += word_distance_matrix[idx_estimated, idx_real] * real_word_at_time[
                idx_estimated, idx_real]

    # If no word in time, difference is calculated from empty string
    for idx_real in range(number_of_real_words):
        word_has_a_match = modelCpp.NewBoolVar(
            'word_has_a_match' + str(idx_real))
        modelCpp.Add(sum([real_word_at_time[idx_estimated, idx_real] for idx_estimated in range(
            number_of_estimated_words)]) == 1).OnlyEnforceIf(word_has_a_match)
        total_phoneme_distance += word_distance_matrix[number_of_estimated_words, idx_real] * word_has_a_match.Not()

    # Loss should be minimized
    modelCpp.Minimize(total_phoneme_distance)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = TIME_THRESHOLD_MAPPING
    status = solver.Solve(modelCpp)

    mapped_indices = []
    try:
        for word_idx in range(number_words):
            mapped_indices.append(
                (solver.Value(estimated_words_order[word_idx])))

        return np.array(mapped_indices, dtype=np.int64)
    except:
        return []


def edit_distance_python(seq1, seq2):
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros((size_x, size_y))
    for x in range(size_x):
        matrix[x, 0] = x
    for y in range(size_y):
        matrix[0, y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x - 1] == seq2[y - 1]:
                matrix[x, y] = min(
                    matrix[x - 1, y] + 1,
                    matrix[x - 1, y - 1],
                    matrix[x, y - 1] + 1
                )
            else:
                matrix[x, y] = min(
                    matrix[x - 1, y] + 1,
                    matrix[x - 1, y - 1] + 1,
                    matrix[x, y - 1] + 1
                )
    print(matrix)
    return (matrix[size_x - 1, size_y - 1])


def get_word_distance_matrix(words_estimated: list, words_real: list) -> np.array:
    number_of_real_words = len(words_real)
    number_of_estimated_words = len(words_estimated)

    word_distance_matrix = np.zeros(
        (number_of_estimated_words + offset_blank, number_of_real_words))
    for idx_estimated in range(number_of_estimated_words):
        for idx_real in range(number_of_real_words):
            word_distance_matrix[idx_estimated, idx_real] = edit_distance_python(
                words_estimated[idx_estimated], words_real[idx_real])

    if offset_blank == 1:
        for idx_real in range(number_of_real_words):
            word_distance_matrix[number_of_estimated_words, idx_real] = len(words_real[idx_real])
    return word_distance_matrix


def get_best_mapped_words(words_estimated: list, words_real: list) -> list:
    word_distance_matrix = get_word_distance_matrix(
        words_estimated, words_real)

    mapped_indices = get_best_path_from_distance_matrix(word_distance_matrix)

    # In case or-tools doesn't converge, go to a faster, low-quality solution
    if len(mapped_indices) == 0:
        mapped_indices = (dtw_from_distance_matrix(word_distance_matrix)).path[:len(words_estimated), 1]

    mapped_words, mapped_words_indices = get_resulting_string(
        mapped_indices, words_estimated, words_real)

    return mapped_words, mapped_words_indices


def matchSampleAndRecordedWords(real_text, recorded_transcript):
    words_estimated = recorded_transcript.split()

    words_real = real_text.split()

    mapped_words, mapped_words_indices = get_best_mapped_words(
        words_estimated, words_real)

    real_and_transcribed_words = []
    real_and_transcribed_words_ipa = []
    for word_idx in range(len(words_real)):
        if word_idx >= len(mapped_words) - 1:
            mapped_words.append('-')
        real_and_transcribed_words.append(
            (words_real[word_idx], mapped_words[word_idx]))
        real_and_transcribed_words_ipa.append((ipa.convert(words_real[word_idx]),
                                               ipa.convert(mapped_words[word_idx])))
    return real_and_transcribed_words, real_and_transcribed_words_ipa, mapped_words_indices


def removePunctuation(word: str) -> str:
    return ''.join([char for char in word if char not in punctuation])


def getPronunciationAccuracy(real_and_transcribed_words) -> tuple[Any, list[float]]:
    total_mismatches = 0.
    number_of_phonemes = 0.
    current_words_pronunciation_accuracy = []
    for pair in real_and_transcribed_words:
        real_without_punctuation = removePunctuation(pair[0]).lower()
        number_of_word_mismatches = edit_distance_python(
            real_without_punctuation, removePunctuation(pair[1]).lower())
        total_mismatches += number_of_word_mismatches
        number_of_phonemes_in_word = len(real_without_punctuation)
        number_of_phonemes += number_of_phonemes_in_word

        current_words_pronunciation_accuracy.append(float(
            number_of_phonemes_in_word - number_of_word_mismatches) / number_of_phonemes_in_word * 100)

    percentage_of_correct_pronunciations = (
                                                   number_of_phonemes - total_mismatches) / number_of_phonemes * 100

    return np.round(percentage_of_correct_pronunciations), current_words_pronunciation_accuracy
