import argparse
import numpy as np
import os
import torch


from PIL import ImageFile
from torchvision import models, datasets, transforms
from torch import optim, nn
from tqdm import tqdm


class DogBreedModelTrainer:
    """The class will instantiate a pretrained model, add a linear layer and train it on the input data.
    The best result is stored in a file and can be loaded and it's performance tested.
    """

    def __init__(
        self,
        n_threads=12,
        model_file_out="hound_model_vgg.pt",
        input_data_basedir=".",
        image_size=224,
        linear_layer_size=500,
        n_epochs=12,
    ):

        ImageFile.LOAD_TRUNCATED_IMAGES = True

        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            print("CUDA is being used to speed up the process.")

        self.n_threads = n_threads
        self.n_epochs = n_epochs
        self.model_file_out = model_file_out
        self.image_size = image_size
        self.linear_layer_size = linear_layer_size
        self.input_data_basedir = input_data_basedir
        self.last_conv_size = int((self.image_size / 2 ** 3) ** 2)

        print("Last conv layer size: {}".format(self.last_conv_size))

        self.normalize_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.train_transform = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.RandomRotation(5),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                self.normalize_transform,
            ]
        )

        self.transform = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                self.normalize_transform,
            ]
        )

        train_dog_data = datasets.ImageFolder(
            os.path.join(self.input_data_basedir, "train"),
            transform=self.train_transform,
        )
        valid_dog_data = datasets.ImageFolder(os.path.join(self.input_data_basedir, "valid"), transform=self.transform)
        test_dog_data = datasets.ImageFolder(os.path.join(self.input_data_basedir, "test"), transform=self.transform)

        num_classes = len(train_dog_data.classes)

        self.train_loader = torch.utils.data.DataLoader(
            train_dog_data, batch_size=20, shuffle=True, num_workers=self.n_threads
        )
        self.valid_loader = torch.utils.data.DataLoader(valid_dog_data, batch_size=20, num_workers=self.n_threads)
        self.test_loader = torch.utils.data.DataLoader(test_dog_data, batch_size=20, num_workers=self.n_threads)

        print(
            "There are {:d} train, {:d} validation and {:d} test images.".format(
                len(train_dog_data), len(valid_dog_data), len(test_dog_data)
            )
        )

        self.model = models.vgg16(pretrained=True)

        for param in self.model.features.parameters():
            param.requires_grad = False

        self.model.classifier[6] = nn.Linear(self.model.classifier[6].in_features, num_classes)

        if self.use_cuda:
            self.model = self.model.cuda()

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.classifier.parameters(), lr=0.001)

    def test(self):
        """A function to test the trained model."""

        # monitor test loss and accuracy
        test_loss = 0.0
        correct = 0.0
        total = 0.0

        self.model.eval()
        for batch_idx, (data, target) in enumerate(self.test_loader):
            # move to GPU
            if self.use_cuda:
                data, target = data.cuda(), target.cuda()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = self.model(data)
            # calculate the loss
            loss = self.criterion(output, target)
            # update average test loss
            test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss.data - test_loss))
            # convert output probabilities to predicted class
            pred = output.data.max(1, keepdim=True)[1]
            # compare predictions to true label
            correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
            total += data.size(0)

        print("Test Loss: {:.6f}\n".format(test_loss))

        print("\nTest Accuracy: %2d%% (%2d/%2d)" % (100.0 * correct / total, correct, total))

    def train(self):
        """Function to train the model for a given number of epochs and show the progress"""

        # initialize tracker for minimum validation loss
        valid_loss_min = np.Inf

        for epoch in tqdm(range(1, self.n_epochs + 1)):
            # initialize variables to monitor training and validation loss
            train_loss = 0.0
            valid_loss = 0.0

            # train the model
            self.model.train()
            for batch_idx, (data, target) in tqdm(enumerate(self.train_loader), total=334):
                # move to GPU
                if self.use_cuda:
                    data, target = data.cuda(), target.cuda()
                ## find the loss and update the model parameters accordingly
                ## record the average training loss, using something like
                ## train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))

            # validate the model
            self.model.eval()
            for batch_idx, (data, target) in enumerate(self.valid_loader):
                # move to GPU
                if self.use_cuda:
                    data, target = data.cuda(), target.cuda()
                ## update the average validation loss
                output = self.model(data)
                loss = self.criterion(output, target)
                valid_loss += loss.item() * data.size(0)

            valid_loss = valid_loss / len(self.valid_loader.sampler)

            # print training/validation statistics
            print("Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}".format(epoch, train_loss, valid_loss))

            if valid_loss < valid_loss_min:
                print("Validation loss decreased {:f} -> {:f}".format(valid_loss_min, valid_loss))
                torch.save(self.model.state_dict(), self.model_file_out)
                valid_loss_min = valid_loss

        self.model.load_state_dict(torch.load(self.model_file_out))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input-data-basedir",
        default=".",
        help="Provide input data base path",
        type=str,
        dest="input_data_basedir",
    )
    parser.add_argument(
        "-m",
        "--model-file-out",
        default=".",
        help="Provide path to where the trained model should be saved",
        type=str,
        dest="model_file_out",
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=12,
        help="Provide the number of epochs for training",
        type=int,
        dest="n_epochs",
    )
    args = parser.parse_args()
    dbmt = DogBreedModelTrainer(
        input_data_basedir=args.input_data_basedir,
        model_file_out=args.model_file_out,
        n_epochs=args.n_epochs,
    )
    dbmt.train()
    dbmt.test()
