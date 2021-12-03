# students_major.py
# predict major from sex, units, state, test_score
# PyTorch 1.7.0-CPU Anaconda3-2020.02  Python 3.7.6
# Windows 10
import manage_csv
import numpy as np
import time
import torch as T
device = T.device("cpu")  # apply to Tensor or Module
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

# -----------------------------------------------------------


class StudentDataset(T.utils.data.Dataset):
  # sex units   state   test_score  major
  # -1  0.395   0 0 1   0.5120      1
  #  1  0.275   0 1 0   0.2860      2
  # -1  0.220   1 0 0   0.3350      0
  # sex: -1 = male, +1 = female
  # state: maryland, nebraska, oklahoma
  # major: finance, geology, history

  def __init__(self, src_file, is_test):
    # all_xy = np.loadtxt(src_file, max_rows=n_rows,
    #                     usecols=[0, 1, 2, 3, 4, 5, 6], delimiter="\t",
    #                     skiprows=0, comments="#", dtype=np.float32)

    x, y, names = manage_csv.load_csv(src_file, "train" in src_file)
    
    if is_test:
      _, x, _, y = train_test_split(x, y, test_size=0.2)
    print(x.shape)
    print(src_file)
    self.x_data = \
        T.tensor(normalize(x), dtype=T.float32).to(device)
    self.y_data = \
        T.tensor(y, dtype=T.int64).to(device)

    # n = len(all_xy)
    # tmp_x = all_xy[0:n, 0:6]  # all rows, cols [0,6)
    # tmp_y = all_xy[0:n, 6]    # 1-D required

    # self.x_data = \
    #     T.tensor(tmp_x, dtype=T.float32).to(device)
    # self.y_data = \
    #     T.tensor(tmp_y, dtype=T.int64).to(device)

  def __len__(self):
    return len(self.x_data)

  def __getitem__(self, idx):
    preds = self.x_data[idx]
    trgts = self.y_data[idx]
    sample = {
        'predictors': preds,
        'targets': trgts
    }
    return sample

# -----------------------------------------------------------


class Net(T.nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.hid1 = T.nn.Linear(34, 10)  # 6-(10-10)-3
    self.hid2 = T.nn.Linear(10, 10)
    self.hid3 = T.nn.Linear(10, 10)
    self.oupt = T.nn.Linear(10, 10)

    T.nn.init.xavier_uniform_(self.hid1.weight)
    T.nn.init.zeros_(self.hid1.bias)
    T.nn.init.xavier_uniform_(self.hid2.weight)
    T.nn.init.zeros_(self.hid2.bias)
    T.nn.init.xavier_uniform_(self.hid3.weight)
    T.nn.init.zeros_(self.hid3.bias)
    T.nn.init.xavier_uniform_(self.oupt.weight)
    T.nn.init.zeros_(self.oupt.bias)

    self.dropout = T.nn.Dropout(0.25)

  def forward(self, x):
    z = T.relu(self.hid1(x))
    z = T.relu(self.hid2(z))
    z = self.dropout(z)
    z = T.relu(self.hid3(z))
    z = self.oupt(z)  # no softmax: CrossEntropyLoss()
    return z

# -----------------------------------------------------------


def accuracy(model, ds):
  # assumes model.eval()
  # granular but slow approach
  n_correct = 0
  n_wrong = 0
  for i in range(len(ds)):
    X = ds[i]['predictors']
    Y = ds[i]['targets']  # [0] [1] or [2]
    with T.no_grad():
      oupt = model(X)  # logits form

    big_idx = T.argmax(oupt)  # [0] [1] or [2]
    
    if big_idx == Y:
      n_correct += 1
    else:
      n_wrong += 1

  acc = (n_correct * 1.0) / (n_correct + n_wrong)
  return acc

# -----------------------------------------------------------


def accuracy_quick(model, dataset):
  # assumes model.eval()
  # en masse but quick
  n = len(dataset)
  X = dataset[0:n]['predictors']
  Y = T.flatten(dataset[0:n]['targets'])  # 1-D

  with T.no_grad():
    oupt = model(X)
  # (_, arg_maxs) = T.max(oupt, dim=1)  # old style
  arg_maxs = T.argmax(oupt, dim=1)  # collapse cols
  num_correct = T.sum(Y == arg_maxs)
  acc = (num_correct * 1.0 / len(dataset))
  return acc.item()

# -----------------------------------------------------------


def main():
  # 0. get started
  print("\nBegin predict student major \n")
  np.random.seed(1)
  T.manual_seed(1)

  # 1. create DataLoader objects
  print("Creating Student Datasets ")

  train_file = "train_data6.csv"
  train_ds = StudentDataset(train_file, False)


  # TODO: proper test
  test_file = "train_data6.csv"
  test_ds = StudentDataset(test_file, True)  # all 40 rows

  bat_size = 10
  train_ldr = T.utils.data.DataLoader(train_ds,
                                      batch_size=bat_size, shuffle=True)

  # 2. create network
  net = Net().to(device)

  # 3. train model
  max_epochs = 1000
  ep_log_interval = 100
  lrn_rate = 0.01

# -----------------------------------------------------------

  loss_func = T.nn.CrossEntropyLoss()  # apply log-softmax()
  optimizer = T.optim.SGD(net.parameters(), lr=lrn_rate)

  print("\nbat_size = %3d " % bat_size)
  print("loss = " + str(loss_func))
  print("optimizer = SGD")
  print("max_epochs = %3d " % max_epochs)
  print("lrn_rate = %0.3f " % lrn_rate)

  print("\nStarting train with saved checkpoints")
  net.train()
  for epoch in range(0, max_epochs):
    T.manual_seed(1 + epoch)  # recovery reproducibility
    epoch_loss = 0  # for one full epoch
    for (batch_idx, batch) in enumerate(train_ldr):
      X = batch['predictors']  # inputs
      Y = batch['targets']     # shape [10,3] (!)

      optimizer.zero_grad()
      oupt = net(X)            # shape [10] (!)

      loss_val = loss_func(oupt, Y)  # avg loss in batch
      epoch_loss += loss_val.item()  # a sum of averages
      loss_val.backward()
      optimizer.step()

    if epoch % ep_log_interval == 0:
      print("epoch = %4d   loss = %0.4f" %
            (epoch, epoch_loss))

      # checkpoint after 0-based epoch 100, 200, etc.
      dt = time.strftime("%Y_%m_%d-%H_%M_%S")
      fn = ".\\Log\\" + str(dt) + str("-") + \
          str(epoch) + "_checkpoint.pt"
# -----------------------------------------------------------

      info_dict = {
          'epoch': epoch,
          'numpy_random_state': np.random.get_state(),
          'torch_random_state': T.random.get_rng_state(),
          'net_state': net.state_dict(),
          'optimizer_state': optimizer.state_dict()
      }
      #T.save(info_dict, fn)

  print("Training complete ")

  # 4. evaluate model accuracy
  print("\nComputing model accuracy")
  net.eval()
  acc_train = accuracy(net, train_ds)  # item-by-item
  print("Accuracy on training data = %0.4f" % acc_train)
  acc_test = accuracy(net, test_ds)  # en masse
  # acc_test = accuracy_quick(net, test_ds)  # en masse
  print("Accuracy on test data = %0.4f" % acc_test)

  # # 5. make a prediction
  # print("\nPredicting for (M  30.5  oklahoma  543): ")
  # inpt = np.array([[-1, 0.305,  0, 0, 1,  0.543]],
  #                 dtype=np.float32)
  # inpt = T.tensor(inpt, dtype=T.float32).to(device)
  # with T.no_grad():
  #   logits = net(inpt)      # values do not sum to 1.0
  # probs = T.softmax(logits, dim=1)  # tensor
  # probs = probs.numpy()  # numpy vector prints better
  # np.set_printoptions(precision=4, suppress=True)
  # print(probs)

  # 6. save model (state_dict approach)
  print("\nSaving trained model ")
  fn = ".\\Models\\student_model.pth"
  #T.save(net.state_dict(), fn)

  # saved_model = Net()
  # saved_model.load_state_dict(T.load(fn))
  # use saved_model to make prediction(s)

  print("\nEnd predict student major demo")


if __name__ == "__main__":
  main()
