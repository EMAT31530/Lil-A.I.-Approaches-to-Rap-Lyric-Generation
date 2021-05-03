# Imports

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.python.summary.summary_iterator import summary_iterator


def collect_accuracy_loss(events_path):
    epoch_loss = []
    epoch_accuracy = []

    for event in summary_iterator(events_path):
        for value in event.summary.value:
            if value.HasField('simple_value'):
                if value.tag == 'epoch_loss':
                    epoch_loss.append(value.simple_value)
                elif value.tag == 'epoch_accuracy':
                    epoch_accuracy.append(value.simple_value)

    return epoch_accuracy, epoch_loss


def plot_loss(train_loss, valid_loss, name, color1='red', color2='green', linestyle='-') -> None:
    epoch_numbers = list(range(len(train_loss)))

    plt.plot(epoch_numbers, valid_loss, c=color1, label='{} valid loss: '.format(name) +
                                                         str(np.round(valid_loss[-1], 3)),
             linestyle=linestyle)
    plt.plot(epoch_numbers, train_loss, c=color2, label='{} train loss'.format(name),
             linestyle=linestyle)

    plt.scatter(epoch_numbers[-1], valid_loss[-1], c='black')
    return None


def plot_accuracy(train_acc, valid_acc, name, color1='red', color2='green', linestyle='-') -> None:
    epoch_numbers = list(range(len(train_acc)))

    plt.plot(epoch_numbers, valid_acc, c=color1, label='{} valid acc: '.format(name) +
                                                       str(np.round(valid_acc[-1], 3)), linestyle=linestyle)
    plt.plot(epoch_numbers, train_acc, c=color2, label='{} train acc'.format(name), linestyle=linestyle)

    plt.scatter(epoch_numbers[-1], valid_acc[-1], c='black')
    return None


def plot_valid_acc(valid_acc, name, color='red', linestyle='-') -> None:
    epoch_numbers = list(range(len(valid_acc)))

    plt.plot(epoch_numbers, valid_acc, c=color, label='{} valid acc: '.format(name) + str(np.round(valid_acc[-1], 3)), linestyle=linestyle)

    plt.scatter(epoch_numbers[-1], valid_acc[-1], c='black')

    return None


def plot_valid_loss(valid_loss, name, color='red', linestyle='-') -> None:
    epoch_numbers = list(range(len(valid_loss)))

    plt.plot(epoch_numbers, valid_loss, c=color, label='{} valid loss: '.format(name) + str(np.round(min(valid_loss), 3)),
             linestyle=linestyle)

    plt.scatter(epoch_numbers[int(np.argmin(valid_loss))], min(valid_loss), c='black')

    return None


if __name__ == "__main__":


    # Check if the model exists:
    # Need to compile model now
    # time_stamp = time.time()
    # tensorboard = TensorBoard(log_dir="logs/{}".format(time_stamp))
    #
    # lstm = CustomModel(x_train, y_train, int_to_word, path="training_82/cp.ckpt".format(time_stamp))
    # lstm.train(x_train, y_train, epochs=200, _tensorboard=tensorboard, batch_size=256, verbose=1)
    # lstm.load()
    #
    # generated_lyrics = generate_rap_lyrics(lstm.model, bars_no_start, word_to_int, int_to_word, padding_length,
    #                                        num_lines=10)
    # print(generated_lyrics)

    # lstm.test(x_test, y_test)
    events_path_train_BEST = 'logs/1618419642.2406116/train/' \
                             'events.out.tfevents.1618419646.DESKTOP-SUN0R45.12996.7016.v2'
    events_path_validation_BEST = 'logs/1618419642.2406116/validation/' \
                             'events.out.tfevents.1618419861.DESKTOP-SUN0R45.12996.4368738.v2'

    # events_path_train_noRegular = 'logs/1617572113.074653/train/events.out.tfevents.1617572117.DESKTOP-SUN0R45.7272.6970.v2'
    # events_path_validation_noRegular = 'logs/1617572113.074653/validation/events.out.tfevents.1617572292.DESKTOP-SUN0R45.7272.4353426.v2'
    #
    # events_path_train_dropout = 'logs/1617625710.6619706/train/events.out.tfevents.1617625721.DESKTOP-SUN0R45.12512.6981.v2'
    # events_path_validation_dropout = 'logs/1617625710.6619706/validation/events.out.tfevents.1617625928.DESKTOP-SUN0R45.12512.4358519.v2'
    #
    # events_path_train_overRegularise = 'logs/1619092192.7243125/train/events.out.tfevents.1619092201.DESKTOP-SUN0R45.18820.7033.v2'
    # events_path_validation_overRegularise = 'logs/1619092192.7243125/validation/events.out.tfevents.1619092416.DESKTOP-SUN0R45.18820.4381233.v2'
    #
    # events_path_train_fastOverRegularise = 'logs/1619128082.2523046/train/events.out.tfevents.1619128088.DESKTOP-SUN0R45.4304.7033.v2'
    # events_path_valdiation_fastOverRegularise = 'logs/1619128082.2523046/validation/events.out.tfevents.1619128311.DESKTOP-SUN0R45.4304.4381233.v2'
    #
    # events_path_train_slowOverRegularise = 'logs/1619172796.5665648/train/events.out.tfevents.1619172813.DESKTOP-SUN0R45.16688.7033.v2'
    # events_path_validation_slowOverRegularise = 'logs/1619172796.5665648/validation/events.out.tfevents.1619173007.DESKTOP-SUN0R45.16688.4381233.v2'

    # events_path_train_onlyDense = 'logs/1619187646.984651/train/events.out.tfevents.1619187652.DESKTOP-SUN0R45.8372.6977.v2'
    # events_path_validation_onlyDense = 'logs/1619187646.984651/validation/events.out.tfevents.1619187845.DESKTOP-SUN0R45.8372.4359440.v2'

    events_path_train_noNorm = 'logs/1619201590.754819/train/events.out.tfevents.1619201595.DESKTOP-SUN0R45.18280.6977.v2'
    events_path_validation_noNorm = 'logs/1619201590.754819/validation/events.out.tfevents.1619201792.DESKTOP-SUN0R45.18280.4356821.v2'

    events_path_train_batchNorm = 'logs/1619209684.23159/train/events.out.tfevents.1619209688.DESKTOP-SUN0R45.16144.7016.v2'
    events_path_validation_batchNorm = 'logs/1619209684.23159/validation/events.out.tfevents.1619209891.DESKTOP-SUN0R45.16144.4368738.v2'

    events_path_train_layerNorm = 'logs/1619215006.231754/train/events.out.tfevents.1619215011.DESKTOP-SUN0R45.22916.7033.v2'
    events_path_validation_layerNorm = 'logs/1619215006.231754/validation/events.out.tfevents.1619215202.DESKTOP-SUN0R45.22916.4369220.v2'

    train_acc1, train_loss1 = collect_accuracy_loss(events_path_train_BEST)
    valid_acc1, valid_loss1 = collect_accuracy_loss(events_path_validation_BEST)
    #
    # train_acc2, train_loss2 = collect_accuracy_loss(events_path_train_noRegular)
    # valid_acc2, valid_loss2 = collect_accuracy_loss(events_path_validation_noRegular)
    #
    # train_acc3, train_loss3 = collect_accuracy_loss(events_path_train_dropout)
    # valid_acc3, valid_loss3 = collect_accuracy_loss(events_path_validation_dropout)
    #
    # train_acc4_0, train_loss4_0 = collect_accuracy_loss(events_path_train_overRegularise)
    # valid_acc4_0, valid_loss4_0 = collect_accuracy_loss(events_path_validation_overRegularise)
    #
    # train_acc4_1, train_loss4_1 = collect_accuracy_loss(events_path_train_fastOverRegularise)
    # valid_acc4_1, valid_loss4_1 = collect_accuracy_loss(events_path_valdiation_fastOverRegularise)
    #
    # train_acc4_2, train_loss4_2 = collect_accuracy_loss(events_path_train_slowOverRegularise)
    # valid_acc4_2, valid_loss4_2 = collect_accuracy_loss(events_path_validation_slowOverRegularise)

    # train_acc5_0, train_loss5_0 = collect_accuracy_loss(events_path_train_onlyDense)
    # valid_acc5_0, valid_loss5_0 = collect_accuracy_loss(events_path_validation_onlyDense)

    train_acc6_0, train_loss6_0 = collect_accuracy_loss(events_path_train_noNorm)
    valid_acc6_0, valid_loss6_0 = collect_accuracy_loss(events_path_validation_noNorm)

    train_acc6_1, train_loss6_1 = collect_accuracy_loss(events_path_train_batchNorm)
    valid_acc6_1, valid_loss6_1 = collect_accuracy_loss(events_path_validation_batchNorm)

    train_acc6_2, train_loss6_2 = collect_accuracy_loss(events_path_train_layerNorm)
    valid_acc6_2, valid_loss6_2 = collect_accuracy_loss(events_path_validation_layerNorm)

    # # =====================================================================================================

    # plt.title('Different accuracies without early stopping for initial models')
    # plt.xlabel('Epoch')
    # plt.ylabel('Accuracy')
    #
    # plot_accuracy(train_acc1, valid_acc1, 'Final model')
    # plot_accuracy(train_acc2, valid_acc2, 'No regularisation', color1='orange', color2='blue', linestyle='--')
    # plot_accuracy(train_acc3, valid_acc3, 'Only dropout', color1="yellow", color2='purple', linestyle='--')
    #
    # plt.legend(loc='upper left')
    #
    # plt.show()

    # # =====================================================================================================

    # plt.title('Different losses (without early stopping)')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    #
    # plot_loss(train_loss1, valid_loss1, 'Final')
    # plot_loss(train_loss2, valid_loss2, 'No regularisation', color1='orange', color2='blue', linestyle='--')
    # plot_loss(train_loss3, valid_loss3, 'Only dropout', color1='yellow', color2='purple', linestyle='--')
    #
    # plt.legend(loc='lower left')
    #
    # plt.show()

    # # =====================================================================================================

    # plt.title('The effects of ridge regularisation on LSTM layers')
    # plt.xlabel('Epoch')
    # plt.ylabel('Accuracy')
    #
    # plot_accuracy(train_acc1, valid_acc1, 'Final model')
    # plot_valid_acc(valid_acc4_0, 'L2 - LR=0.001', color='orange', linestyle='--')
    # plot_valid_acc(valid_acc4_1, 'L2 - LR=0.01', color='yellow', linestyle='--')
    # plot_valid_acc(valid_acc4_2, 'L2 - LR=0.0001', color='pink', linestyle='--')
    #
    # plt.legend(loc='upper right')
    #
    # plt.show()

    # # =====================================================================================================

    # plt.title('The effects of ridge regularisation on LSTM layers')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    #
    # plot_accuracy(train_loss1, valid_loss1, 'Final model')
    # plot_valid_loss(valid_loss4_0, 'L2 - LR=0.001', color='orange', linestyle='--')
    # plot_valid_loss(valid_loss4_1, 'L2 - LR=0.01', color='yellow', linestyle='--')
    # plot_valid_loss(valid_loss4_2, 'L2 - LR=0.0001', color='pink', linestyle='--')
    #
    # plt.legend(loc='upper right')
    #
    # plt.show()

    # # =====================================================================================================

    # plt.title('Only regularising the dense layer')
    # plt.xlabel('Epoch')
    # plt.ylabel('Accuracy')
    #
    # plot_accuracy(train_acc1, valid_acc1, 'Final model')
    # plot_accuracy(train_acc5_0, valid_acc5_0, 'L2 Dense LR=0.001', color1='orange', color2='blue', linestyle='--')
    # plot_valid_acc(valid_acc4_0, 'L2 LR=0.001', color='purple', linestyle='--')
    #
    # plt.legend(loc='upper right')
    #
    # plt.show()

    # # =====================================================================================================

    plt.title('The effect of normalising the input data on validation accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    plot_valid_acc(valid_acc1, 'Final Model', color='red')
    plot_valid_acc(valid_acc6_0, 'No Norm', color='orange', linestyle='--')
    plot_valid_acc(valid_acc6_1, 'Batch Norm', color='green', linestyle='--')
    plot_valid_acc(valid_acc6_2, 'Layer Norm', color='blue', linestyle='--')

    plt.legend(loc='upper left')

    plt.show()

    # =====================================================================================================

    plt.title('The effect of normalising the input data on validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plot_valid_loss(valid_loss1, 'Final Model', color='red')
    plot_valid_loss(valid_loss6_0, 'No Norm', color='orange', linestyle='--')
    plot_valid_loss(valid_loss6_1, 'Batch Norm', color='green', linestyle='--')
    plot_valid_loss(valid_loss6_2, 'Layer Norm', color='blue', linestyle='--')

    plt.legend(loc='upper left')

    plt.show()
