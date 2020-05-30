import torch
from torch import nn
import torch.optim as optim
from torchsummary import summary

import numpy as np

import data
import textrnn

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def main():
	# ======================
	# 超参数
	# ======================
	CELL = "bi-lstm"            # rnn, bi-rnn, gru, bi-gru, lstm, bi-lstm
	BATCH_SIZE = 64
	EMBED_SIZE = 128
	HIDDEN_DIM = 256
	NUM_LAYERS = 1
	CLASS_NUM = 2
	DROPOUT_RATE = 0.0
	EPOCH = 200
	LEARNING_RATE = 0.01
	SAVE_EVERY = 5

	all_var = locals()
	print()
	for var in all_var:
		if var != "var_name":
			print("{0:15}   ".format(var), all_var[var])
	print()

	# ======================
	# 数据
	# ======================
	with open('rt-polaritydata/rt-polarity.pos', 'r', encoding='Windows-1252') as f:
		raw_pos = f.read().split("\n")
	raw_pos = list(filter(lambda x: x not in ['', None], raw_pos))
	with open('rt-polaritydata/rt-polarity.neg', 'r', encoding='Windows-1252') as f:
		raw_neg = f.read().split("\n")
	raw_neg = list(filter(lambda x: x not in ['', None], raw_neg))
	data_helper = data.DataHelper([raw_pos, raw_neg], use_label=True)

	# ======================
	# 构建模型
	# ======================
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = textrnn.TextRNN(
		cell=CELL,
		vocab_size=data_helper.vocab_size,
		embed_size=EMBED_SIZE,
		hidden_dim=HIDDEN_DIM,
		num_layers=NUM_LAYERS,
		class_num=CLASS_NUM,
		dropout_rate=DROPOUT_RATE
	)
	model.to(device)
	summary(model, (20,))
	criteration = nn.CrossEntropyLoss()
	optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
	print()

	# ======================
	# 训练与测试
	# ======================
	for epoch in range(EPOCH):
		generator_train = data_helper.train_generator(BATCH_SIZE)
		generator_test = data_helper.test_generator(BATCH_SIZE)
		train_loss = []
		train_acc = []
		while True:
			try:
				text, label = generator_train.__next__()
			except:
				break
			optimizer.zero_grad()
			y = model(torch.from_numpy(text).long().to(device))
			loss = criteration(y, torch.from_numpy(label).long().to(device))
			loss.backward()
			optimizer.step()
			train_loss.append(loss.item())
			y = y.cpu().detach().numpy()
			train_acc += [1 if np.argmax(y[i]) == label[i] else 0 for i in range(len(y))]

		test_loss = []
		test_acc = []
		while True:
			with torch.no_grad():
				try:
					text, label = generator_test.__next__()
				except:
					break
				y = model(torch.from_numpy(text).long().to(device))
				loss = criteration(y, torch.from_numpy(label).long().to(device))
				test_loss.append(loss.item())
				y = y.cpu().numpy()
				test_acc += [1 if np.argmax(y[i]) == label[i] else 0 for i in range(len(y))]

		print('epoch {:d}   training loss {:.4f}    test loss {:.4f}    train acc {:.4f}    test acc {:.4f}'
		      .format(epoch + 1, np.mean(train_loss), np.mean(test_loss), np.mean(train_acc), np.mean(test_acc)))

		if (epoch + 1) % SAVE_EVERY == 0:
			print('saving parameters')
			os.makedirs('models', exist_ok=True)
			torch.save(model.state_dict(), 'models/textrnn-' + str(epoch) + '.pkl')


if __name__ == '__main__':
	main()
