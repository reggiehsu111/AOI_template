class Trainer:
    def __init__(self, config):
        print(config)
        self.config = config
        self.device = torch.device("cuda:0" if torch.cuda.is_available() and config['use_gpu'] else "cpu")
        self.categories = config['categories']
        self.data_dir = config['data_dir']
        self.DataSet = ECGDataset(data_dir=self.data_dir, categories=self.categories,
                             normalize=True, print_statistics=False)

        # Randomly split dataset into training, validation and test sets
        train_size = int(config['train_val_test'][0]*len(self.DataSet))
        val_size = int(config['train_val_test'][1]*len(self.DataSet))
        test_size = len(self.DataSet) - train_size - val_size

        # random_split() is from PyTorch
        train_DataSet, val_DataSet, test_DataSet = random_split(self.DataSet, [train_size, val_size, test_size])

        self.train_DataLoader = DataLoader(train_DataSet, batch_size=config['batch_size'], shuffle=True, num_workers=4)
        self.val_DataLoader = DataLoader(val_DataSet, batch_size=config['batch_size'], shuffle=True, num_workers=4)
        self.test_DataLoader = DataLoader(test_DataSet, batch_size=config['batch_size'], shuffle=True, num_workers=4)
        

        self.model = Net()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['learning_rate'])
        self.criterion = nn.CrossEntropyLoss(reduction='sum')
        self.checkpoint_dir = config['checkpoint_dir']
        if not os.path.exists(self.checkpoint_dir):
            print("{} not exists, create one!".format(self.checkpoint_dir))
            os.makedirs(self.checkpoint_dir)

        if config['azure_log']:
            # log to Azure
            from azureml.core.run import Run
            self.azure_run = Run.get_context()
            self.azure_run.log('batch_size', np.float(config['batch_size']))
            self.azure_run.log('learning_rate', np.float(config['learning_rate']))
        else:
            # log to Tensorboard
            from tensorboardX import SummaryWriter
            self.arch = type(self.model).__name__
            self.writer = SummaryWriter(comment=self.arch)
            self.writer.add_graph(self.model, (torch.Tensor(1,1,3750)))
    
    def train(self):
        self.model.to(self.device)
        best_validate_acc = 0
        saved_model_testing_acc = 0
        for epoch in range(1, self.config['num_epochs']+1):
            # Train
            sys.stdout.write("\n")
            sys.stdout.flush()
            
            self.model.train()
            for batch_idx, (x, y) in enumerate(self.train_DataLoader):
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                x = x.view((-1,1,3750))
                outputs = self.model(x)
                loss = self.criterion(outputs, y)
                loss /= y.size(0)
                # print('y.shape:', y.shape)
                loss.backward()
                self.optimizer.step()
                sys.stdout.write("\rTrain epoch {}: [{}/{} ({:.0f}%)] Loss: {:.6f}".format(
                    epoch, batch_idx*y.size(0), len(self.train_DataLoader)*y.size(0), 100*batch_idx/len(self.train_DataLoader), loss.cpu().item()))
                sys.stdout.flush()
            
            # Eval: get train, val and test accuracies
            training_acc, training_loss = self._compute_metrics(self.train_DataLoader)
            validate_acc, validation_loss = self._compute_metrics(self.val_DataLoader)
            testing_acc, testing_loss = self._compute_metrics(self.test_DataLoader)
            sys.stdout.write(" train acc: {:.3f}, valid acc: {:.3f}, test acc: {:.3f}".format(
                training_acc, validate_acc, testing_acc))
            
            # if epoch % self.config['save_per_epochs'] == 0:
            if validate_acc > best_validate_acc:
                best_validate_acc = validate_acc
                saved_model_testing_acc = testing_acc
                self._save_checkpoint(epoch, training_loss)

            if self.config['azure_log']:
                self.azure_run.log('best_validate_acc', best_validate_acc)
                self.azure_run.log('saved_model_testing_acc', saved_model_testing_acc)
                self.azure_run.log('train_acc', training_acc)
                self.azure_run.log('train_loss', training_loss)
                self.azure_run.log('valid_acc', validate_acc)
                self.azure_run.log('valid_loss', validation_loss)
                self.azure_run.log('test_acc', testing_acc)
                self.azure_run.log('test_loss', testing_loss)
            else:
                self.writer.add_scalar('data/train_acc', training_acc, epoch)
                self.writer.add_scalar('data/train_loss', training_loss, epoch)
                self.writer.add_scalar('data/valid_acc', validate_acc, epoch)
                self.writer.add_scalar('data/valid_loss', validation_loss, epoch)
                self.writer.add_scalar('data/test_acc', testing_acc, epoch)
                self.writer.add_scalar('data/test_loss', testing_loss, epoch)

        print()

    def _save_checkpoint(self, epoch, loss):
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'config': self.config
        }
        # filename = os.path.join(self.checkpoint_dir, 'checkpoint-epoch{:03d}-loss-{:.4f}.pth.tar'
        #     .format(epoch, loss))
        filename = os.path.join(self.checkpoint_dir, 'best_model.pth')
        torch.save(state, filename)

    def _compute_metrics(self, input_DataLoader):
        # torch.no_grad() saves memory and increases computation speed
        with torch.no_grad():
            self.model.eval()
            total_metrics = 0
            total_loss = 0
            for batch_idx, (x, y) in enumerate(input_DataLoader):
                x, y = x.to(self.device), y.to(self.device)
                x = x.view((-1,1,3750))
                outputs = self.model(x)
                loss = self.criterion(outputs, y)
                outputs_class = np.argmax(outputs.cpu().numpy(), axis=1)
                acc = accuracy(outputs_class, y.cpu().numpy())
                total_metrics += acc
                total_loss += loss.cpu().item()
            return total_metrics / len(input_DataLoader.dataset),  total_loss / len(input_DataLoader.dataset)
