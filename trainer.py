import torch
import numpy as np
import tqdm as tqdm

from utils.stats_manager import StatsManager
from utils.data_logs import save_logs_train, save_logs_eval
import os


class Trainer:
    def __init__(self, network, train_dataloader, eval_dataloader, criterion, optimizer,
                 lr_scheduler, logs_writer, config, discriminator_struct, criterionD):
        self.config = config
        self.network = network
        self.stats_manager = StatsManager(config)
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.logs_writer = logs_writer

        self.discriminator_struct = discriminator_struct
        self.criterionD = criterionD

        self.best_metric = 0.0

    def compute_generator_loss(self, predictions_fake):
        label = torch.ones(len(predictions_fake[0])).to(self.config['device'])

        # Since we just updated D, perform another forward pass of all-fake batch through D
        err_g = 0.
        for key in self.discriminator_struct:

            output = self.discriminator_struct[key]['disc'](
                **{"xl": predictions_fake[2],
                   "xm": predictions_fake[1],
                   "xs": predictions_fake[0]}).view(-1)
            # Calculate G's loss based on this output
            errG = self.criterionD(output, label)
            err_g = err_g + errG

        return err_g

    def compute_discriminator_loss(self, predictions_fake, labels_real):

        for key in self.discriminator_struct:
            self.discriminator_struct[key]['optim'].zero_grad()

            # Forward pass real batch through D
            label_real = torch.ones(len(predictions_fake[0])).to(self.config['device'])
            input_D = {"xs": labels_real[key][0], "xm": labels_real[key][1], "xl": labels_real[key][2]}

            output_real = self.discriminator_struct[key]['disc'](**input_D).view(-1)
            # Calculate loss on all-real batch
            errD_real = self.criterionD(output_real, label_real)

            ## Train with all-fake batch
            # Generate batch of latent vectors
            label_fake = torch.zeros(len(predictions_fake[0])).to(self.config['device'])
            # Classify all fake batch with D
            output_fake = self.discriminator_struct[key]['disc'](
                **{"xl": predictions_fake[2],
                   "xm": predictions_fake[1],
                   "xs": predictions_fake[0]}).view(-1)

            # Calculate D's loss on the all-fake batch
            errD_fake = self.criterionD(output_fake.detach(), label_fake)
            # Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake
            errD.backward()

            # Update D
            self.discriminator_struct[key]['optim'].step()

    def train_epoch(self, epoch):
        running_loss = []
        self.network.train()
        for idx, (inputs, labels) in enumerate(self.train_dataloader, 0):
            inputs = inputs.to(self.config['device']).float()
            for key in labels.keys():
                for i in range(0, len(labels[key])):
                    labels[key][i] = labels[key][i].to(self.config['device']).float()

            predictions = self.network(inputs)

            if epoch >= self.config['start_adv_epoch']:
                self.compute_discriminator_loss(predictions, labels)

            loss_task = self.criterion(predictions, labels, self.config['teachers_weights'])

            if epoch >= self.config['start_adv_epoch']:
                loss_adversarial = self.compute_generator_loss(predictions)
                loss = loss_task + self.config['adv_gen_weight'] * loss_adversarial
            else:
                loss = loss_task

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if epoch >= self.config['start_adv_epoch']:
                self.compute_discriminator_loss(predictions, labels)

            running_loss.append(loss.item())
            if idx % self.config['print_loss'] == 0:
                running_loss = np.mean(np.array(running_loss))
                print(f'Training loss on iteration {idx} = {running_loss}')
                save_logs_train(os.path.join(self.config['exp_path'], self.config['exp_name']),
                                f'Training loss on iteration {idx} = {running_loss}')

                self.logs_writer.add_scalar('Training Loss', running_loss, epoch * len(self.train_dataloader) + idx)
                running_loss = []

    @torch.no_grad()
    def eval_net(self, epoch):
        stats_labels = []
        stats_predictions_s = []
        stats_predictions_m = []
        stats_predictions_l = []
        stats_predictions_mix = []

        stats_video = []
        stats_frame = []

        running_eval_loss = 0.0
        self.network.eval()
        print("Evaluare")
        for inputs, labels, true_labels, video, frame in tqdm.tqdm(self.eval_dataloader):
            inputs = inputs.to(self.config['device']).float()

            for key in labels.keys():
                for i in range(0, len(labels[key])):
                    labels[key][i] = labels[key][i].to(self.config['device']).float()

            predictions = self.network(inputs)

            eval_loss = self.criterion(predictions, labels, self.config['teachers_weights'])
            running_eval_loss += eval_loss.item()

            stats_predictions_s.append(predictions[0].detach().cpu().numpy().max(-1))
            stats_predictions_m.append(predictions[1].detach().cpu().numpy().max((-3, -2, -1)))
            stats_predictions_l.append(predictions[2].detach().cpu().numpy())

            stats_predictions_mix.append(
                self.mix_predictions(predictions[0], predictions[1], predictions[2])
            )

            stats_labels.append(true_labels.detach().cpu().numpy())
            stats_video.append(video)
            stats_frame.append(frame.detach().cpu().numpy())

        # Small
        micro_auc, macro_auc = self.stats_manager.get_stats_v2(predictions_raw=stats_predictions_s,
                                                               labels=stats_labels,
                                                               video=stats_video, frame=stats_frame)
        running_eval_loss = running_eval_loss / len(self.eval_dataloader)

        print(
            f'### Evaluation loss on epoch {epoch} = {running_eval_loss}, :::S::: Micro-AUC = {micro_auc}, Macro-AUC = {macro_auc}')
        save_logs_eval(os.path.join(self.config['exp_path'], self.config['exp_name']),
                       f'::S:: Evaluation loss on epoch {epoch} = {running_eval_loss}, Micro-AUC = {micro_auc}, Macro-AUC = {macro_auc}')

        # Medium
        micro_auc, macro_auc = self.stats_manager.get_stats_v2(predictions_raw=stats_predictions_m,
                                                               labels=stats_labels,
                                                               video=stats_video, frame=stats_frame)

        print(
            f'### Evaluation loss on epoch {epoch} = {running_eval_loss}, :::M::: Micro-AUC = {micro_auc}, Macro-AUC = {macro_auc}')
        save_logs_eval(os.path.join(self.config['exp_path'], self.config['exp_name']),
                       f'::M:: Evaluation loss on epoch {epoch} = {running_eval_loss}, Micro-AUC = {micro_auc}, Macro-AUC = {macro_auc}')

        # Large
        micro_auc, macro_auc, rbdc, tbdc = self.stats_manager.get_stats_v2(predictions_raw=stats_predictions_l,
                                                                           labels=stats_labels,
                                                                           video=stats_video, frame=stats_frame,
                                                                           rbdc_tbdc=True)
        running_eval_loss = running_eval_loss / len(self.eval_dataloader)

        print(
            f'### Evaluation loss on epoch {epoch} = {running_eval_loss}, :::L::: Micro-AUC = {micro_auc}, Macro-AUC = {macro_auc}')
        save_logs_eval(os.path.join(self.config['exp_path'], self.config['exp_name']),
                       f'::L:: Evaluation loss on epoch {epoch} = {running_eval_loss}, Micro-AUC = {micro_auc}, Macro-AUC = {macro_auc},'
                       f'RBDC = {0}, TBDC = {0}')

        # Mix
        micro_auc, macro_auc = self.stats_manager.get_stats_v2(predictions_raw=stats_predictions_mix,
                                                               labels=stats_labels,
                                                               video=stats_video, frame=stats_frame)

        print(
            f'### Evaluation loss on epoch {epoch} = {running_eval_loss}, :::MIX::: Micro-AUC = {micro_auc}, Macro-AUC = {macro_auc}')
        save_logs_eval(os.path.join(self.config['exp_path'], self.config['exp_name']),
                       f'::MIX:: Evaluation loss on epoch {epoch} = {running_eval_loss}, Micro-AUC = {micro_auc}, Macro-AUC = {macro_auc}')

        if self.best_metric < micro_auc:
            self.best_metric = micro_auc
            self.save_net_state(None, best=True)

        self.logs_writer.add_scalar('Validation Loss', running_eval_loss, (epoch + 1) * len(self.train_dataloader))

    def train(self):
        if self.config['resume_training'] is True:
            checkpoint = torch.load(os.path.join(self.config['exp_path'],
                                                 self.config['exp_name'],
                                                 'latest_checkpoint.pkl'),
                                    map_location=self.config['device'])
            self.network.load_state_dict(checkpoint['model_weights'])

        for i in range(1, self.config['train_epochs'] + 1):
            print('Training on epoch ' + str(i))
            self.train_epoch(i)
            self.save_net_state(i, latest=True)

            if i % self.config['eval_net_epoch'] == 0:
                self.eval_net(i)

            if i % self.config['save_net_epochs'] == 0:
                self.save_net_state(i)

            self.lr_scheduler.step()

    def save_net_state(self, epoch, latest=False, best=False):
        if latest is True:
            path_to_save = os.path.join(self.config['exp_path'], self.config['exp_name'], f'latest_checkpoint.pkl')
            to_save = {
                'epoch': epoch,
                'model_weights': self.network.state_dict(),
                'optimizer': self.optimizer.state_dict()
            }
            torch.save(to_save, path_to_save)
        elif best is True:
            path_to_save = os.path.join(self.config['exp_path'], self.config['exp_name'], f'best_model.pkl')
            to_save = {
                'epoch': epoch,
                'stats': self.best_metric,
                'model_weights': self.network.state_dict()
            }
            torch.save(to_save, path_to_save)
        else:
            path_to_save = os.path.join(self.config['exp_path'], self.config['exp_name'], f'model_epoch_{epoch}.pkl')
            to_save = {
                'epoch': epoch,
                'model_weights': self.network.state_dict()
            }
            torch.save(to_save, path_to_save)

    def test_net(self, test_dataloader):
        running_eval_loss = 0.0
        predictions_stats = []
        labels_stats = []

        checkpoint = torch.load(os.path.join(self.config['exp_path'], self.config['exp_name'], 'best_model.pkl'),
                                map_location=self.config['device'])
        network = checkpoint['model']
        network.eval()
        for idx, (inputs, labels) in enumerate(test_dataloader, 0):
            inputs = inputs.to(self.config['device']).float()
            labels = labels.to(self.config['device']).long()

            with torch.no_grad():
                predictions = network(inputs)
            loss = self.criterion(predictions, labels, self.config['teachers_weights'])

            predictions_stats.append(predictions.detach().cpu().numpy())
            labels_stats.append(labels.detach().cpu().numpy())

            running_eval_loss += loss.item()

        predictions_stats = np.hstack(predictions_stats)
        labels_stats = np.hstack(labels_stats)

        stats = StatsManager(self.config)
        performance = stats.get_stats(predictions_stats, labels_stats)

        running_eval_loss = running_eval_loss / len(test_dataloader)

        print(f'Test loss = {running_eval_loss} Performance = {performance}')

        history = open(os.path.join(self.config['exp_path'], self.config['exp_name'], '__testStats__.txt'), "a")
        history.write(f'Test loss = {running_eval_loss} Performance = {performance}')
        history.close()

    def view_data(self, epoch):
        self.network.eval()
        checkpoint = torch.load(os.path.join(self.config['exp_path'],
                                             self.config['exp_name'],
                                             'best_model.pkl'),
                                map_location=self.config['device'])
        self.network.load_state_dict(checkpoint['model_weights'])
        import matplotlib.pyplot as plt

        for idx, (inputs, labels, _, _, _) in enumerate(self.train_dataloader, 0):
            inputs = inputs.to(self.config['device']).float()
            labels = labels.to(self.config['device']).float()

            predictions = self.network(inputs)

            inputs = np.clip(127 + inputs.permute(0, 3, 2, 1).detach().cpu().numpy() * 128, a_min=0, a_max=255).astype(
                np.int)
            labels = labels.detach().cpu().numpy()
            predictions = predictions[0].detach().cpu().numpy()

            plt.figure("Input")
            plt.imshow(inputs[0])

            plt.figure("Label")
            plt.imshow(labels[0, 0].T)

            plt.figure("Prediction")
            plt.imshow(predictions[0, 0].T)

            plt.show()

    def mix_predictions(self, pred_s, pred_m, pred_l):
        pred_s = pred_s.detach().cpu().numpy().max(-1)
        pred_m = pred_m.detach().cpu().numpy().max((-3, -2, -1))
        pred_l = pred_l.detach().cpu().numpy().max((-3, -2, -1))

        return np.mean((pred_s, pred_m, pred_l), 0)