"""
This code is partially based on the repository of https://github.com/locuslab/fast_adversarial (Wong et al., ICLR'20)

python train_our.py --dataset=cifar10 --epochs=30 --lr_max=0.1 --n_final_eval=1000 --train_alpha 10 --random_start --batch_size 128 --gpu 5 --exact --num-samples 10

python train_our.py --dataset=cifar10 --epochs=30 --lr_max=0.1 --n_final_eval=1000 --train_alpha 10 --random_start --batch_size 128 --gpu 6
"""
import argparse
import os
import time
import numpy as np
from scipy.linalg import eigh
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import utils
import data
import models

import torch.autograd.forward_ad as fwAD

from datetime import datetime
from utils import rob_acc

try:
	import apex.amp as amp
except:
	pass

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--batch_size', default=128, type=int)
	parser.add_argument('--data_dir', default=None, type=str)
	parser.add_argument('--dataset', default='cifar10', choices=['mnist', 'svhn', 'cifar10', 'cifar10_binary', 'cifar10_binary_gs',
																 'uniform_noise', 'imagenet'], type=str)
	parser.add_argument('--model', default='resnet18', choices=['resnet18', 'lenet', 'cnn'], type=str)
	parser.add_argument('--epochs', default=30, type=int,
						help='15 epochs to reach 45% adv acc, 30 epochs to reach the reported clean/adv accs')
	parser.add_argument('--lr_schedule', default='cyclic', choices=['cyclic', 'piecewise'])
	parser.add_argument('--lr_max', default=0.2, type=float, help='0.05 in Table 1, 0.2 in Figure 2')
	parser.add_argument('--eps', default=8.0, type=float)
	parser.add_argument('--attack_iters', default=10, type=int, help='n_iter of pgd for evaluation')
	parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay aka l2 regularization')
	parser.add_argument('--fname', default='spat_cifar10', type=str)
	parser.add_argument('--seed', default=0, type=int)
	parser.add_argument('--gpu', default=0, type=int)
	parser.add_argument('--debug', action='store_true')
	parser.add_argument('--half_prec', action='store_true', help='if enabled, runs everything as half precision')
	parser.add_argument('--eval_early_stopped_model', action='store_true', help='whether to evaluate the model obtained via early stopping')
	parser.add_argument('--eval_iter_freq', default=200, type=int, help='how often to evaluate test stats')
	parser.add_argument('--n_eval_every_k_iter', default=256, type=int, help='on how many examples to eval every k iters')
	parser.add_argument('--n_layers', default=1, type=int, help='#layers on each conv layer (for model == cnn)')
	parser.add_argument('--n_filters_cnn', default=16, type=int, help='#filters on each conv layer (for model==cnn)')
	parser.add_argument('--batch_size_eval', default=256, type=int, help='batch size for the final eval with pgd rr; 6 GB memory is consumed for 1024 examples with fp32 network')
	parser.add_argument('--n_final_eval', default=-1, type=int, help='on how many examples to do the final evaluation; -1 means on all test examples.')

	parser.add_argument('--n_train_alpha_warmup_epochs', default=0, type=int)
	parser.add_argument('--num-samples', default=1, type=int)
	parser.add_argument('--train_alpha', default=None, type=float)
	parser.add_argument('--exact', action='store_true')
	parser.add_argument('--fast', action='store_true')
	parser.add_argument('--random_start', action='store_true')
	parser.add_argument('--dist-type', default='radamecher', type=str, choices=['radamecher', 'gaussian', 'orthogonal-gaussian', 'eigen'])
	return parser.parse_args()


def main():
	args = get_args()
	os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

	if not args.data_dir:
		if os.path.isdir('/data/LargeData/cifar/'):
			args.data_dir = '/data/LargeData/cifar/'

	cur_timestamp = str(datetime.now())[:-3]  # we also include ms to prevent the probability of name collision
	model_width = {'linear': '', 'cnn': args.n_filters_cnn, 'lenet': '', 'resnet18': ''}[args.model]
	model_str = '{}{}'.format(args.model, model_width)
	model_name = '{} dataset={} model={} eps={} epochs={} lr_max={} seed={}'.format(
		cur_timestamp, args.dataset, model_str, args.eps, args.epochs, args.lr_max, args.seed)
	if not os.path.exists('models'):
		os.makedirs('models')
	logger = utils.configure_logger(model_name, args.debug)
	logger.info(args)
	half_prec = args.half_prec
	n_cls = 2 if 'binary' in args.dataset else 10

	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	torch.cuda.manual_seed(args.seed)

	n_eval_every_k_iter = args.n_eval_every_k_iter
	args.pgd_alpha = args.eps / 4
	if args.train_alpha is None:
		args.train_alpha = args.eps

	eps, pgd_alpha, train_alpha= args.eps / 255, args.pgd_alpha / 255, args.train_alpha / 255
	train_data_augm = False if args.dataset in ['mnist'] else True
	train_batches = data.get_loaders(args.data_dir, args.dataset, -1, args.batch_size, train_set=True, shuffle=True, data_augm=train_data_augm)
	train_batches_fast = data.get_loaders(args.data_dir, args.dataset, n_eval_every_k_iter, args.batch_size, train_set=True, shuffle=False, data_augm=False)
	test_batches = data.get_loaders(args.data_dir, args.dataset, args.n_final_eval, args.batch_size_eval, train_set=False, shuffle=False, data_augm=False)
	test_batches_fast = data.get_loaders(args.data_dir, args.dataset, n_eval_every_k_iter, args.batch_size_eval, train_set=False, shuffle=False, data_augm=False)

	model = models.get_model(args.model, n_cls, half_prec, data.shapes_dict[args.dataset], args.n_filters_cnn).cuda()
	model.apply(utils.initialize_weights)
	model.train()

	if args.model == 'resnet18':
		opt = torch.optim.SGD(model.parameters(), lr=args.lr_max, momentum=0.9, weight_decay=args.weight_decay)
	elif args.model == 'cnn':
		opt = torch.optim.Adam(model.parameters(), lr=args.lr_max, weight_decay=args.weight_decay)
	elif args.model == 'lenet':
		opt = torch.optim.Adam(model.parameters(), lr=args.lr_max, weight_decay=args.weight_decay)
	else:
		raise ValueError('decide about the right optimizer for the new model')

	if half_prec:
		model, opt = amp.initialize(model, opt, opt_level="O1")

	lr_schedule = utils.get_lr_schedule(args.lr_schedule, args.epochs, args.lr_max)
	loss_function = nn.CrossEntropyLoss()

	eigenvecs = get_eigenvecs_of_grads(args, model, loss_function, train_batches, eps, times=3)

	train_acc_pgd_best, best_state_dict = 0.0, copy.deepcopy(model.state_dict())
	start_time = time.time()
	time_train, iteration, best_iteration = 0, 0, 0
	for epoch in range(args.epochs + 1):
		train_loss, train_loss_benign, train_acc, train_n = 0, 0, 0, 0
		for i, (X, y) in enumerate(train_batches):
			time_start_iter = time.time()
			# epoch=0 runs only for one iteration (to check the training stats at init)
			if epoch == 0 and i > 0:
				break
			X, y = X.cuda(), y.cuda()
			lr = lr_schedule(epoch - 1 + (i + 1) / len(train_batches))  # epoch - 1 since the 0th epoch is skipped
			opt.param_groups[0].update(lr=lr)

			if args.n_train_alpha_warmup_epochs > 0:
				n_iterations_max_train_alpha = args.n_train_alpha_warmup_epochs * data.shapes_dict[args.dataset][0] // args.batch_size
				alpha_ = min(iteration / n_iterations_max_train_alpha * train_alpha, train_alpha)
			else:
				alpha_ = train_alpha

			## SPAT
			if args.exact:
				if args.random_start:
					noise = torch.empty_like(X).uniform_(-eps, eps)
				else:
					noise = torch.empty_like(X).zero_()
				X_noise = X.add(noise).clamp_(0, 1)

				deltas = get_batch_perturbations(args.dist_type, X.shape[0], args.num_samples, np.prod(X.shape[1:]), eigenvecs)
				grads = []
				for sample_i in range(args.num_samples):
					with torch.no_grad():
						with fwAD.dual_level():
							output, Jdelta = fwAD.unpack_dual(model(fwAD.make_dual(X_noise, deltas[:, sample_i].view_as(X))))
						R = output.softmax(-1) - F.one_hot(y, n_cls)
						grads.append(deltas[:, sample_i].view_as(X) * (R * Jdelta).sum(-1)[:, None, None, None])

				loss_benign = loss_function(output, y)
				noise.add_((sum(grads)/args.num_samples).sign(), alpha=alpha_).clamp_(-eps, eps)
				X_adv = X.add_(noise).clamp_(0, 1)
				loss = loss_function(model(X_adv), y)
			else:
				delta.uniform_(-1, 1).sign_()
				if args.random_start:
					X.add_(torch.empty_like(X).uniform_(-eps, eps)).clamp_(0, 1)

				with fwAD.dual_level():
					output, Jdelta = fwAD.unpack_dual(model(fwAD.make_dual(X, delta)))
				if args.fast:
					Jdelta.detach_()

				R = output.softmax(-1) - F.one_hot(y, n_cls)
				output_adv = output + alpha_ * (R * Jdelta).sum(-1, keepdim=True).sign() * Jdelta
				loss = loss_function(output_adv, y)
				loss_benign = loss_function(output.detach(), y)

			if epoch != 0:
				opt.zero_grad()
				utils.backward(loss, opt, half_prec)
				opt.step()

			time_train += time.time() - time_start_iter
			train_loss += loss.item() * y.size(0)
			train_loss_benign += loss_benign.item() * y.size(0)
			train_acc += (output.max(1)[1] == y).sum().item()
			train_n += y.size(0)

			if iteration % args.eval_iter_freq == 0:
				train_loss = train_loss / train_n
				train_loss_benign = train_loss_benign / train_n
				train_acc = train_acc / train_n

				# it'd be incorrect to recalculate the BN stats on the test sets and for clean / adversarial points
				utils.model_eval(model, half_prec)

				test_acc_clean, _, _ = rob_acc(test_batches_fast, model, eps, pgd_alpha, opt, half_prec, 0, 1)
				test_acc_fgsm, test_loss_fgsm, fgsm_deltas = rob_acc(test_batches_fast, model, eps, eps, opt, half_prec, 1, 1, rs=False)
				test_acc_pgd, test_loss_pgd, pgd_deltas = rob_acc(test_batches_fast, model, eps, pgd_alpha, opt, half_prec, args.attack_iters, 1)
				cos_fgsm_pgd = utils.avg_cos_np(fgsm_deltas, pgd_deltas)
				train_acc_pgd, _, _ = rob_acc(train_batches_fast, model, eps, pgd_alpha, opt, half_prec, args.attack_iters, 1)  # needed for early stopping

				grad_x = utils.get_grad_np(model, test_batches_fast, eps, opt, half_prec, rs=False)
				grad_eta = utils.get_grad_np(model, test_batches_fast, eps, opt, half_prec, rs=True)
				cos_x_eta = utils.avg_cos_np(grad_x, grad_eta)

				time_elapsed = time.time() - start_time
				train_str = '[train] loss {:.3f}, loss benign {:.3f}, acc {:.2%} acc_pgd {:.2%}'.format(train_loss, train_loss_benign, train_acc, train_acc_pgd)
				test_str = '[test] acc_clean {:.2%}, acc_fgsm {:.2%}, acc_pgd {:.2%}, cos_x_eta {:.3}, cos_fgsm_pgd {:.3}'.format(
					test_acc_clean, test_acc_fgsm, test_acc_pgd, cos_x_eta, cos_fgsm_pgd)
				logger.info('{}-{}: {}  {} ({:.2f}m, {:.2f}m)'.format(epoch, iteration, train_str, test_str,
																	  time_train/60, time_elapsed/60))

				if train_acc_pgd > train_acc_pgd_best:  # catastrophic overfitting can be detected on the training set
					best_state_dict = copy.deepcopy(model.state_dict())
					train_acc_pgd_best, best_iteration = train_acc_pgd, iteration

				utils.model_train(model, half_prec)
				train_loss, train_loss_benign, train_acc, train_n = 0, 0, 0, 0

			iteration += 1

		if epoch == args.epochs:
			torch.save({'last': model.state_dict(), 'best': best_state_dict}, 'models/{} epoch={}.pth'.format(model_name, epoch))
			# disable global conversion to fp16 from amp.initialize() (https://github.com/NVIDIA/apex/issues/567)
			context_manager = amp.disable_casts() if half_prec else utils.nullcontext()
			with context_manager:
				last_state_dict = copy.deepcopy(model.state_dict())
				half_prec = False  # final eval is always in fp32
				model.load_state_dict(last_state_dict)
				utils.model_eval(model, half_prec)
				opt = torch.optim.SGD(model.parameters(), lr=0)

				attack_iters, n_restarts = (50, 10) if not args.debug else (10, 3)
				test_acc_clean, _, _ = rob_acc(test_batches, model, eps, pgd_alpha, opt, half_prec, 0, 1)
				test_acc_pgd_rr, _, deltas_pgd_rr = rob_acc(test_batches, model, eps, pgd_alpha, opt, half_prec, attack_iters, n_restarts)
				logger.info('[last: test on 10k points] acc_clean {:.2%}, pgd_rr {:.2%}'.format(test_acc_clean, test_acc_pgd_rr))

				if args.eval_early_stopped_model:
					model.load_state_dict(best_state_dict)
					utils.model_eval(model, half_prec)
					test_acc_clean, _, _ = rob_acc(test_batches, model, eps, pgd_alpha, opt, half_prec, 0, 1)
					test_acc_pgd_rr, _, deltas_pgd_rr = rob_acc(test_batches, model, eps, pgd_alpha, opt, half_prec, attack_iters, n_restarts)
					logger.info('[best: test on 10k points][iter={}] acc_clean {:.2%}, pgd_rr {:.2%}'.format(
						best_iteration, test_acc_clean, test_acc_pgd_rr))

		utils.model_train(model, half_prec)

	logger.info('Done in {:.2f}m'.format((time.time() - start_time) / 60))


def get_batch_perturbations(dist_type, batch_size, num_samples, dim, eigenvecs=None):
	if dist_type == 'radamecher':
		deltas = torch.empty(batch_size, num_samples, dim, device='cuda')
		deltas.uniform_(-1, 1).sign_()
	elif dist_type == 'gaussian':
		deltas = torch.empty(batch_size, num_samples, dim, device='cuda')
		deltas.normal_()
	elif dist_type == 'orthogonal-gaussian':
		deltas = torch.empty(batch_size, dim, num_samples, device='cuda')
		deltas.normal_()
		q, _ = torch.linalg.qr(deltas)
		deltas = q.permute(0, 2, 1)
		deltas /= np.sqrt(num_samples / dim)
		# with np.printoptions(precision=2, suppress=True):
		#     print((deltas[0] @ deltas[0].T).data.cpu().numpy())
		#     print((deltas[0].T @ deltas[0]).data.cpu().numpy())
	elif dist_type == 'eigen':
		deltas = torch.empty(batch_size, eigenvecs.shape[1], num_samples, device='cuda')
		deltas.normal_()
		deltas /= (deltas ** 2).sum(1, keepdim=True).sqrt()
		deltas = torch.einsum("dk,bks->bsd", eigenvecs, deltas)
		# with np.printoptions(precision=2, suppress=True):
		#     print((deltas[0].T @ deltas[0]).data.cpu().numpy())
	else:
		raise NotImplementedError
	return deltas

def get_eigenvecs_of_grads(args, model, loss_function, train_batches, eps, times=1, th=0.5):
	grads = []
	for i, (X, y) in tqdm(enumerate(train_batches), desc='Collecting grads', total=len(train_batches)):
		X, y = X.cuda(), y.cuda()

		for _ in range(times):
			if args.random_start:
				noise = torch.empty_like(X).uniform_(-eps, eps)
			else:
				noise = torch.empty_like(X).zero_()
			X_noise = X.add(noise).clamp_(0, 1)
			X_noise.requires_grad_()

			loss = loss_function(model(X_noise), y) * y.size(0)
			grad = torch.autograd.grad(loss, X_noise)[0]
			grads.append(grad.flatten(1).cpu())

	grads = torch.cat(grads)

	grads -= grads.mean(0)
	cov = grads.T @ grads / grads.shape[0]
	p, q = eigh(cov.cpu().numpy())
	p = torch.from_numpy(p).float()[range(-1, -(p.shape[0]+1), -1)].cuda()
	q = torch.from_numpy(q).float()[:, range(-1, -(p.shape[0]+1), -1)].cuda()
	for i in range(1, p.shape[0]):
		if p[:i].sum()/p.sum() > th:
			break

	p = p[:i]
	q = q[:, :i]

	print(i, q.shape)

	with np.printoptions(precision=3, suppress=True):
		print(p.data.cpu().numpy())
		# print((q.T @ q).data.cpu().numpy())

	return q

if __name__ == "__main__":
	main()
