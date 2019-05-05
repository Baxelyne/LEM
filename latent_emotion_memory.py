# coding: utf-8
from keras import backend as K
from keras.layers import Input, Dense, Lambda, Activation, Dropout, Flatten, Bidirectional, Embedding, dot, Concatenate, \
    TimeDistributed
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.utils import Progbar
from keras.layers.recurrent import GRU
from keras.layers.merge import add, concatenate
import numpy as np
from sklearn.metrics import accuracy_score
from data_preparating import get_data
from utils import *


def main():
    ######################## parameters ########################


    EMBED_DIM = 200
    HIDDEN_NUM = [300, 300, 300]
    EMOTION_EMB_DIM = 200
    MAX_SEQ_LEN = 64
    EMOTION_DISTRIBUTION_LENGTH = 200

    BATCH_SIZE = 16
    MAX_EPOCH = 1000
    MIN_EPOCH = 10

    PATIENT = 5
    PATIENT_GLOBAL = 30
    TARGET_SPARSITY = 0.75
    KL_GROWING_EPOCH = 0

    ######################## data input ########################

    bow_train, bow_test, seq_train, seq_test, label_train, label_test, \
    dictionary_seq, dictionary_bow, label_dict = get_data("sample_cn.csv", "stopwords_cn.txt")

    EMOTION_NUM = len(label_dict)  # emotion number
    CATEGORY = 2  # binary

    seq_train_pad = pad_sequences(seq_train, maxlen=MAX_SEQ_LEN)
    seq_test_pad = pad_sequences(seq_test, maxlen=MAX_SEQ_LEN)

    bow_train, bow_train_ind = generate_arrays_from_source(bow_train)
    bow_test, bow_test_ind = generate_arrays_from_source(bow_test)
    test_count_indices = np.sum(bow_test_ind, axis=1)

    ######################## LEM ########################
    ## latent emotion module

    ebow_input = Input(shape=(len(dictionary_bow),), name="ebow_input")

    e1 = Dense(HIDDEN_NUM[0], activation='relu', name='encoder_1')
    e2 = Dense(HIDDEN_NUM[1], activation='relu', name='encoder_2')
    e3 = Dense(EMOTION_NUM, name='encoder_3_mean')
    e4 = Dense(EMOTION_NUM, name='encoder_3_var')
    h = e1(ebow_input)
    h = e2(h)

    es = Dense(HIDDEN_NUM[2], use_bias=False, name='encoder_shortcut')
    h = add([h, es(ebow_input)])

    z_mean = e3(h)
    z_log_var = e4(h)

    def sampling(args):
        mu, log_sigma = args
        epsilon = K.random_normal(shape=(EMOTION_NUM,), mean=0.0, stddev=1.0)
        return mu + K.exp(log_sigma / 2) * epsilon

    hidden = Lambda(sampling, output_shape=(EMOTION_NUM,))([z_mean, z_log_var])

    g1 = Dense(EMOTION_NUM, activation="tanh", name='decoder_1')
    g2 = Dense(EMOTION_NUM, activation="tanh", name='decoder_2')
    g3 = Dense(EMOTION_NUM, activation="tanh", name='decoder_3')
    g4 = Dense(EMOTION_NUM, name='decoder_4')

    def generate(h):
        tmp = g1(h)
        tmp = g2(tmp)
        tmp = g3(tmp)
        tmp = g4(tmp)
        r = add([Activation("tanh")(tmp), h])

        return r

    represent = generate(hidden)

    l1_strength = CustomizedL1L2(l1=0.001)
    d1 = Dense(len(dictionary_bow), activation="softmax", kernel_regularizer=l1_strength, name="p_x_given_h")
    p_x_given_h = d1(represent)

    represent_mu = generate(z_mean)

    def kl_loss(x_true, x_decoded):
        kl_term = - 0.5 * K.sum(
            1 - K.square(z_mean) + z_log_var - K.exp(z_log_var),
            axis=-1)
        return kl_term

    def nnl_loss(x_true, x_decoder):
        nnl_term = - K.sum(x_true * K.log(x_decoder + 1e-32), axis=-1)
        return nnl_term

    kl_strength = K.variable(1.0)

    le_module = Model(ebow_input, [represent_mu, p_x_given_h])
    le_module.compile(loss=[kl_loss, nnl_loss], loss_weights=[kl_strength, 1.0], optimizer="adagrad")

    ## memory module

    text_input = Input(shape=(MAX_SEQ_LEN,), dtype='int32', name='seq_input')
    memo_input = Input(shape=(EMOTION_NUM,), dtype='int32', name="psudo_input")

    seq_emb = Embedding(len(dictionary_seq) + 1,
                        EMBED_DIM,
                        weights=None,
                        input_length=MAX_SEQ_LEN,
                        trainable=True)
    emotion_emb = Embedding(EMOTION_NUM, len(dictionary_bow), input_length=EMOTION_NUM, trainable=False)

    c1 = Dense(EMOTION_EMB_DIM, activation='relu', name='seq_encoder')
    t1 = Dense(EMOTION_EMB_DIM, activation='relu', name='emo_w_encoder')
    out_dense = TimeDistributed(Dense(CATEGORY, activation='softmax', name='classifier'))

    x = seq_emb(text_input)
    x = c1(x)
    x = Dropout(0.05)(x)

    wt_emb = emotion_emb(memo_input)
    wt_emb = t1(wt_emb)

    ## transforming distribution into seperate representation
    emo_dis1 = Dense(MAX_SEQ_LEN, activation='relu', name='emo_dis1')
    emo_dis2 = Dense(MAX_SEQ_LEN, activation='relu', name='emo_dis2')
    emo_dis3 = Dense(MAX_SEQ_LEN, activation='relu', name='emo_dis3')
    emo_dis4 = Dense(MAX_SEQ_LEN, activation='relu', name='emo_dis4')
    emo_dis5 = Dense(MAX_SEQ_LEN, activation='relu', name='emo_dis5')
    emo_dis6 = Dense(MAX_SEQ_LEN, activation='relu', name='emo_dis6')
    emo_dis7 = Dense(MAX_SEQ_LEN, activation='relu', name='emo_dis7')
    emo_dis8 = Dense(MAX_SEQ_LEN, activation='relu', name='emo_dis8')
    represent_emo1 = emo_dis1(represent_mu)
    represent_emo2 = emo_dis2(represent_mu)
    represent_emo3 = emo_dis3(represent_mu)
    represent_emo4 = emo_dis4(represent_mu)
    represent_emo5 = emo_dis5(represent_mu)
    represent_emo6 = emo_dis6(represent_mu)
    represent_emo7 = emo_dis7(represent_mu)
    represent_emo8 = emo_dis8(represent_mu)

    def define_N_module(input):
        x, wt_emb, represent_mu = input
        f1 = Dense(EMOTION_EMB_DIM, activation="relu", name='f1')
        o1 = Dense(EMOTION_EMB_DIM, activation='relu', name='o1')
        s1 = Dense(EMOTION_EMB_DIM * 2, activation='relu', name='s1')

        match = dot([x, wt_emb], axes=(2, 2))
        represent_mu = K.expand_dims(represent_mu, axis=-1)
        joint_match = add([represent_mu, match], )
        joint_match = f1(joint_match)

        emotion_sum = add([joint_match, x])
        emotion_sum = o1(emotion_sum)
        rep = Flatten()(emotion_sum)
        rep = Dropout(0.05, )(rep)
        rep = s1(rep)
        rep = K.expand_dims(rep, axis=1)

        return rep

    concated_arr = [
        Lambda(define_N_module, name="module_0")([x, wt_emb, represent_emo1]),
        Lambda(define_N_module, name="module_1")([x, wt_emb, represent_emo2]),
        Lambda(define_N_module, name="module_2")([x, wt_emb, represent_emo3]),
        Lambda(define_N_module, name="module_3")([x, wt_emb, represent_emo4]),
        Lambda(define_N_module, name="module_4")([x, wt_emb, represent_emo5]),
        Lambda(define_N_module, name="module_5")([x, wt_emb, represent_emo6]),
        Lambda(define_N_module, name="module_6")([x, wt_emb, represent_emo7]),
        Lambda(define_N_module, name="module_7")([x, wt_emb, represent_emo8]),
    ]

    concated_out = Concatenate(axis=1)(concated_arr)
    out_fusion = Bidirectional(GRU(EMOTION_EMB_DIM, return_sequences=True))(concated_out)
    concated_out = concatenate([out_fusion, concated_out], axis=2)
    out = Dropout(0.1)(concated_out)

    cls_out = out_dense(out)

    em_module = Model([ebow_input, text_input, memo_input], cls_out)
    em_module.compile(optimizer="adam", loss='categorical_crossentropy', metrics=["accuracy"])

    ##################### training ###########################

    num_batches = int(bow_train.shape[0] / BATCH_SIZE)
    kl_base = float(KL_GROWING_EPOCH * num_batches)
    optimize_ntm = True
    min_bound_ntm = np.inf
    max_test_acc = - np.inf
    epoch_since_improvement = 0
    epoch_since_improvement_global = 0

    # training
    for epoch in range(1, MAX_EPOCH + 1):
        progress_bar = Progbar(target=num_batches)
        epoch_train = []
        epoch_test = []

        # shuffle data
        indices = np.arange(bow_train.shape[0])
        np.random.shuffle(indices)
        seq_train_shuffle = seq_train_pad[indices]
        bow_train_shuffle = bow_train[indices]
        bow_train_ind_shuffle = bow_train_ind[indices]
        label_train_shuffle = label_train[indices]
        psudo_indices = np.expand_dims(np.arange(EMOTION_NUM), axis=0)
        psudo_train = np.repeat(psudo_indices, seq_train_pad.shape[0], axis=0)
        psudo_test = np.repeat(psudo_indices, seq_test_pad.shape[0], axis=0)

        if optimize_ntm:
            print('Epoch {}/{} training {}'.format(epoch, MAX_EPOCH, "latent emotion module"))
            for index in range(num_batches):
                if epoch < KL_GROWING_EPOCH:
                    K.set_value(kl_strength, np.float32((epoch * num_batches + index) / kl_base))
                else:
                    K.set_value(kl_strength, 1.)
                bow_batch = bow_train_shuffle[index * BATCH_SIZE:(index + 1) * BATCH_SIZE]
                bow_index_batch = bow_train_ind_shuffle[index * BATCH_SIZE:(index + 1) * BATCH_SIZE]
                epoch_train.append(le_module.train_on_batch(
                    bow_batch, [np.zeros([len(bow_batch), EMOTION_NUM]), bow_index_batch]))
                progress_bar.update(index + 1)

            [train_loss, train_kld, train_nnl] = np.mean(epoch_train, axis=0)
            print("LE module training loss: %.4f (kld: %.4f, nnl: %.4f)" % (train_loss, train_kld, train_nnl))
            sparsity = check_sparsity(le_module)
            update_l1(l1_strength, sparsity, TARGET_SPARSITY)

            print('\nvalidating on test set')
            val_loss, kld, nnl = le_module.evaluate(bow_test, [bow_test, bow_test_ind], verbose=0)
            bound = np.exp(val_loss / np.mean(test_count_indices))
            print("estimated perplexity upper bound on validation set: %.3f" % bound)

            if bound < min_bound_ntm and epoch > KL_GROWING_EPOCH:
                print("New best val bound: %.3f in %d epoch\n" % (bound, epoch))
                min_bound_ntm = bound
                epoch_since_improvement = 0
                epoch_since_improvement_global = 0
            elif bound >= min_bound_ntm:
                epoch_since_improvement += 1
                epoch_since_improvement_global += 1
                print("No improvement in epoch %d\n" % epoch)

            if epoch < KL_GROWING_EPOCH:
                print("Growing kl strength %.3f" % K.get_value(kl_strength))

            if epoch_since_improvement > PATIENT and epoch > MIN_EPOCH:
                optimize_ntm = False
                epoch_since_improvement = 0
                beta_exp = np.exp(le_module.get_weights()[-2])
                beta = beta_exp / (np.sum(beta_exp, 1)[:, np.newaxis])
                emotion_emb.set_weights([beta])
            if epoch_since_improvement_global > PATIENT_GLOBAL:
                break

        else:
            print('Epoch {}/{} training {}'.format(epoch, MAX_EPOCH, "emotion memory module"))
            for index in range(num_batches):
                bow_batch = bow_train_shuffle[index * BATCH_SIZE:(index + 1) * BATCH_SIZE]
                seq_batch = seq_train_shuffle[index * BATCH_SIZE:(index + 1) * BATCH_SIZE]
                psudo_batch = psudo_train[index * BATCH_SIZE:(index + 1) * BATCH_SIZE]
                label_batch = label_train_shuffle[index * BATCH_SIZE:(index + 1) * BATCH_SIZE]
                epoch_train.append(em_module.train_on_batch(
                    [bow_batch, seq_batch, psudo_batch], label_batch))
                progress_bar.update(index + 1)
            train_loss, train_acc = np.mean(epoch_train, axis=0)
            print("LEM loss: %.4f" % (train_loss))

            y_pred = em_module.predict([bow_test, seq_test_pad, psudo_test])
            y_pred_label = np.argmax(y_pred, axis=2)
            y_true_label = np.argmax(label_test, axis=2)
            test_acc = accuracy_score(y_true_label, y_pred_label)

            if test_acc > max_test_acc:
                max_test_acc = test_acc
                print("New best acc on val in %d epoch: %.4f\n" % (epoch, test_acc))
                epoch_since_improvement = 0
                epoch_since_improvement_global = 0

            else:
                epoch_since_improvement += 1
                epoch_since_improvement_global += 1
                print("No improvement in epoch %d, acc: %.4f\n" % (epoch, test_acc))

            if epoch_since_improvement > PATIENT and epoch > MIN_EPOCH:
                optimize_ntm = True
                epoch_since_improvement = 0
            if epoch_since_improvement_global > PATIENT_GLOBAL:
                break
    print('best acc: %.4f' % max_test_acc)


if __name__ == "__main__":
    main()
