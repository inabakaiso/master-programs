import numpy as np


# BLUEによる評価
def BLEU_evaluation(out_filename, ref_filename, BLEU_num):
    with open(out_filename, encoding='shift_jis') as f:
        out_all = [s.strip() for s in f.readlines()]

    with open(ref_filename, encoding="utf-8_sig") as f:
        ref_all = [s[26:].strip() for s in f.readlines()]

    BLEU_total = 0
    for i in range(len(out_all)):
        # 出力文取得
        out = []
        for n in range(BLEU_num):
            if n == 0:
                out.append(out_all[i].split())
            else:
                out.append([])
                for j in range(len(out[0]) - n):
                    out[n].append(out[0][j:j + n + 1])

        # 参照文取得
        ref = []
        for n in range(BLEU_num):
            ref.append([])
            for j in range(5):
                if n == 0:
                    ref[n].append(ref_all[i * 5 + j].split())
                else:
                    ref[n].append([])
                    for k in range(len(ref[0][j]) - n):
                        ref[n][j].append(ref[0][j][k:k + n + 1])

        # 合計語数取得
        out_total = len(out[0])
        ref_total = 0
        diff = 100
        for j in range(5):
            ref_temp = len(ref[0][j])
            if abs(ref_temp - out_total) < diff:
                diff = abs(ref_temp - out_total)
                ref_total = ref_temp
            elif abs(ref_temp - out_total) == diff:
                if ref_temp < out_total:
                    ref_total = ref_temp

        # 一致数取得 ※一致した要素を削除していく
        match = [0] * BLEU_num
        temp = [0] * BLEU_num
        for n in range(BLEU_num):
            for out_word in out[n]:
                temp[n] = 0
                for j in range(5):
                    for ref_word in ref[n][j]:
                        if out_word == ref_word:
                            temp[n] = 1
                            ref[n][j].remove(out_word)
                            break
                match[n] += temp[n]

        # 画像一枚に対するBLEU算出，全体に加算
        BP = min(1, np.exp(1 - ref_total / out_total))
        p_temp = 0
        for n in range(BLEU_num):
            if match[n] != 0:
                p_temp += 1 / BLEU_num * np.log(match[n] / len(out[n]))
            else:
                p_temp = -1000
                break
        BLEU_temp = BP * np.exp(p_temp)
        if p_temp == -1000:
            BLEU_temp = 0
        BLEU_total += BLEU_temp

    BLEU = BLEU_total / len(out_all)
    print("BLEU" + str(BLEU_num) + ":", end='')
    print(BLEU)


def BLEU_evaluation4(out_filename, ref_filename):
    for i in range(4):
        BLEU_evaluation(out_filename, ref_filename, i + 1)


def lcs(X, Y, m, n):
    global lcs_mem
    if lcs_mem[m][n] != 0:
        return lcs_mem[m][n]
    elif m == 0 or n == 0:
        return 0
    elif X[m - 1] == Y[n - 1]:
        lcs_mem[m][n] = 1 + lcs(X, Y, m - 1, n - 1)
    else:
        lcs_mem[m][n] = max(lcs(X, Y, m, n - 1), lcs(X, Y, m - 1, n))
    return lcs_mem[m][n]


def ROUGE_evaluation(out_filename, ref_filename):  # ROUGE-Lによる評価
    with open(out_filename, encoding='shift-jis') as f:
        out_all = [s.strip() for s in f.readlines()]

    with open(ref_filename, encoding="utf-8_sig") as f:
        ref_all = [s[26:].strip() for s in f.readlines()]

    ROUGE_total = 0
    beta = 1.2
    for i in range(len(out_all)):
        # 出力文取得
        out = out_all[i].split()

        # 参照文取得
        ref = []
        for j in range(5):
            ref.append(ref_all[i * 5 + j].split())

        # 合計語数取得
        out_total = len(out)
        precision = 0
        recall = 0
        ROUGE_total_temp = 0
        for j in range(5):
            ref_total = len(ref[j])

            # lcsに入れる前に処理
            xxx = []
            yyy = []
            for a_word in out:
                if a_word in ref[j]:
                    xxx.append(a_word)
            for b_word in ref[j]:
                if b_word in out:
                    yyy.append(b_word)
            if len(xxx) > 30:  # 長すぎる文を制限(同じ単語列の繰り返しになってることが多い)
                xxx = xxx[:30]
            global lcs_mem
            lcs_mem = [[0 for j in range(31)] for i in range(31)]
            con = lcs(xxx, yyy, len(xxx), len(yyy))

            recall = con / ref_total
            precision = con / out_total
            if recall + precision != 0:
                ROUGE_temp = (1 + beta * beta) * recall * precision / (recall + beta * beta * precision)
                if ROUGE_total_temp < ROUGE_temp:
                    ROUGE_total_temp = ROUGE_temp

        ROUGE_total += ROUGE_total_temp

    ROUGE = ROUGE_total / len(out_all)
    print("ROUGE-L: ", end='')
    print(ROUGE)


def CIDEr_evaluation(out_filename, ref_filename):  # CIDErによる評価
    with open(ref_filename, encoding='utf-8_sig') as f:
        ref_all = [s[26:].strip() for s in f.readlines()]

    I = int(len(ref_all) / 5)  # 画像枚数
    N = 4

    vocab = []
    # 参照文 IDF計算
    ref_len = int(len(ref_all) / 5)
    for n in range(N):  # n:n-gram数
        vocab.append({})
        for i in range(ref_len):  # i番目の画像
            dup = []
            for j in range(5):  # i番目の画像のj番目の文
                ref0 = ref_all[i * 5 + j].split()
                if n == 0:
                    for word in ref0:
                        if word not in vocab[n]:
                            vocab[n][word] = 1
                            dup.append(word)
                        else:
                            if word not in dup:
                                vocab[n][word] += 1
                                dup.append(word)
                else:
                    ref = []
                    for k in range(len(ref0) - n):
                        ref.append(ref0[k:k + n + 1])
                    for word in ref:
                        word_link = ''
                        for link in word:
                            word_link += link + '/'
                        if word_link not in vocab[n]:
                            vocab[n][word_link] = 1
                            dup.append(word_link)
                        else:
                            if word_link not in dup:
                                vocab[n][word_link] += 1
                                dup.append(word_link)
    IDF = []
    for n in range(N):
        IDF.append({})
        for word in vocab[n].keys():
            IDF[n][word] = np.log(I / (vocab[n][word] + 1))  # +1するか否か

    with open(out_filename, encoding='shift-jis') as f:
        out_all = [s.strip() for s in f.readlines()]

    # 出力文取得
    out_TF = []  # [n][i]{}
    for n in range(N):
        out_TF.append([])
        for i in range(len(out_all)):
            out_TF[n].append({})
            out0 = out_all[i].split()
            if n == 0:
                for word in out0:
                    if word not in out_TF[n][i]:
                        out_TF[n][i][word] = 1
                    else:
                        out_TF[n][i][word] += 1
                out_TF[n][i][word] /= len(out0)
            else:
                out = []
                for j in range(len(out0) - n):
                    out.append(out0[j:j + n + 1])
                for word in out:
                    word_link = ''
                    for link in word:
                        word_link += link + '/'
                    if word_link not in out_TF[n][i]:
                        out_TF[n][i][word_link] = 1
                    else:
                        out_TF[n][i][word_link] += 1
                    out_TF[n][i][word_link] /= len(out)

    # 参照文取得
    ref_TF = []  # [n][i]{}
    for n in range(N):
        ref_TF.append([])
        for i in range(len(ref_all)):
            ref_TF[n].append({})
            ref0 = ref_all[i].split()
            if n == 0:
                for word in ref0:
                    if word not in ref_TF[n][i]:
                        ref_TF[n][i][word] = 1
                    else:
                        ref_TF[n][i][word] += 1
                ref_TF[n][i][word] /= len(ref0)
            else:
                ref = []
                for j in range(len(ref0) - n):
                    ref.append(ref0[j:j + n + 1])
                for word in ref:
                    word_link = ''
                    for link in word:
                        word_link += link + '/'
                    if word_link not in ref_TF[n][i]:
                        ref_TF[n][i][word_link] = 1
                    else:
                        ref_TF[n][i][word_link] += 1
                    ref_TF[n][i][word_link] /= len(ref)

    sigma = 6
    CIDEr = [0] * I
    for i in range(I):
        for n in range(N):
            CIDEr_n = 0
            for j in range(5):
                out_g = 0
                ref_g = 0
                dot_g = 0
                for word in out_TF[n][i].keys():
                    if word in IDF[n]:
                        out_g += (out_TF[n][i][word] * IDF[n][word]) * (out_TF[n][i][word] * IDF[n][word])
                    else:
                        out_g += (out_TF[n][i][word] * np.log(I)) * (out_TF[n][i][word] * np.log(I))
                    if word in ref_TF[n][i * 5 + j]:
                        if word in IDF[n]:
                            ref_g += (ref_TF[n][i * 5 + j][word] * IDF[n][word]) * (
                                        ref_TF[n][i * 5 + j][word] * IDF[n][word])
                        else:
                            ref_g += (ref_TF[n][i * 5 + j][word] * np.log(I)) * (ref_TF[n][i * 5 + j][word] * np.log(I))
                    if word in ref_TF[n][i * 5 + j]:
                        if word in IDF[n]:
                            dot_g += min((out_TF[n][i][word] * IDF[n][word]),
                                         (ref_TF[n][i * 5 + j][word] * IDF[n][word])) \
                                     * (ref_TF[n][i * 5 + j][word] * IDF[n][word])
                        else:
                            dot_g += min((out_TF[n][i][word] * np.log(I)), (ref_TF[n][i * 5 + j][word] * np.log(I))) \
                                     * (ref_TF[n][i * 5 + j][word] * np.log(I))
                out_g = np.sqrt(out_g)
                ref_g = np.sqrt(ref_g)
                len_diff = (len(out_all[i].split()) - n) - (len(ref_all[i * 5 + j].split()) - n)
                if out_g != 0 and ref_g != 0:
                    CIDEr_n += np.exp(-len_diff * len_diff / (2 * sigma * sigma)) * dot_g / (out_g * ref_g)  # 0除算防ぐ
            CIDEr_n *= 10 / 5  # 10/m
            CIDEr[i] += CIDEr_n
        CIDEr[i] /= N

    avg_CIDEr = 0
    for i in range(I):
        avg_CIDEr += CIDEr[i]
    avg_CIDEr /= I

    print("CIDEr-D: ", end='')
    print(avg_CIDEr)