import argparse
import logging
import sys
import numpy
import time
from collections import OrderedDict
import torchvision.models as models
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import random_split
from tqdm import tqdm
from data import *
from helpers import *
from utils import AverageMeter, accuracy

seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

logger = logging.getLogger(__name__)
best_acc = 0
best_acc_valid = 0

def get_num_classes(dataset_name):
    if dataset_name == 'pacs':
        return 7
    elif dataset_name == 'vlcs':
        return 5
    elif dataset_name == 'office_home':
        return 65
    else:
        raise ValueError('Invalid dataset name: {}'.format(dataset_name))


def create_model(args, dataset_name):
    if args.arch == 'resnet18':    
        model = models.resnet18(weights='ResNet18_Weights.DEFAULT')
    elif args.arch == 'resnet50':
        model = models.resnet50(weights='ResNet50_Weights.DEFAULT')
    
    num_ftrs = model.fc.in_features
    num_classes = get_num_classes(args.dataset_name)
    model.fc = nn.Linear(num_ftrs, num_classes)
    print("Dataset: ", args.dataset_name) 
    print("Number of classes: ", num_classes)
    print("Model Architecture: ", args.arch)
    return model


def main():
    parser = argparse.ArgumentParser(description='PyTorch FixMatch Training')

    parser.add_argument('--num-workers', type=int, default=8,
                        help='number of workers')
    parser.add_argument('--dataset_name', default='pacs', type=str,
                        choices=['pacs', 'vlcs','office_home'],
                        help='dataset name')

    # number of labeled training examples
    parser.add_argument('--num-labeled', type=int, default=210,
                        help='number of labeled data')

    parser.add_argument("--expand-labels", action="store_true",
                        help="expand labels to fit eval steps")

    parser.add_argument('--arch', default='resnet18', type=str,
                        choices=['resnet18', 'resnet50'])
    # Epochs = total steps / eval steps
    #35840
    parser.add_argument('--total-steps', default=512*20, type=int,
                        help='number of total steps to run')
    #896
    parser.add_argument('--eval-step', default=512, type=int,
                        help='number of eval steps to run')

    parser.add_argument('--start-epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts)')

    parser.add_argument('--batch-size', default=64, type=int,
                        help='train batchsize')
    parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                        help='initial learning rate')
    # warmup steps to have a very low learning rate at initial steps (warm up steps)
    parser.add_argument('--warmup', default=0, type=float,
                        help='warmup epochs (unlabeled data based)')
    # Small value to avoid overfitting
    parser.add_argument('--wdecay', default=5e-4, type=float,
                        help='weight decay')

    #efault = True
    parser.add_argument('--nesterov', action='store_true', default=True,
                       help='use nesterov momentum')

    parser.add_argument('--use-ema', action='store_true', default=True,
                        help='use EMA model')

    parser.add_argument('--ema-decay', default=0.999, type=float,
                        help='EMA decay rate')
    ##Unlabeled Batch size
    parser.add_argument('--mu', default=7, type=int,
                        help='coefficient of unlabeled batch size')
    ## Loss Coefficient
    parser.add_argument('--lambda-u', default=1, type=float,
                        help='coefficient of unlabeled loss')

    parser.add_argument('--T', default=1, type=float,
                        help='pseudo label temperature')

    parser.add_argument('--threshold', default=0.95, type=float,
                        help='pseudo label threshold')
    ## Change directory
    parser.add_argument('--out', default='./',
                        help='directory to output the result')

    parser.add_argument('--resume', default='', type=str,
                        help='path to latest checkpoint (default: none)')

    parser.add_argument("--amp", action="store_true",
                        help="use 16-bit (mixed) precision through NVIDIA apex AMP")

    parser.add_argument("--opt_level", type=str, default="O1",
                        help="apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                        "See details at https://nvidia.github.io/apex/amp.html")

    parser.add_argument('--no-progress', action='store_true',
                        help="don't use progress bar")

    args = parser.parse_args()
    global best_acc, best_acc_valid

    if args.arch == 'resnet50':
        filename = 'logs_resnet50.log'
    else:
        filename = 'logs_resnet18.log'
    
    logging.basicConfig(filename=args.out + '/' + filename, level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')
        

    device = torch.device('cuda')
    args.device = device

    logger.info(dict(args._get_kwargs()))

    # Forming Data Loaders
    labeled_dataset, unlabeled_dataset, validation_dataset, test_dataset =  load_dataset(args.dataset_name)   
    train_sampler = RandomSampler 

    # train and unlabeled loaders
    labeled_trainloader = DataLoader(
        labeled_dataset,
        sampler=train_sampler(labeled_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True)

    unlabeled_trainloader = DataLoader(
        unlabeled_dataset,
        sampler=train_sampler(unlabeled_dataset),
        batch_size=args.batch_size*args.mu,
        num_workers=args.num_workers,
        drop_last=True)
    
    # Test and Validation
    test_loader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers
        )
    
    val_loader = DataLoader(
        validation_dataset,
        sampler=SequentialSampler(validation_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers
        )
    # create model and move to device
    model = create_model(args, args.dataset_name)
    model.to(args.device)

    no_decay = ['bias', 'bn']
    grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': args.wdecay},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}]

    optimizer = optim.SGD(grouped_parameters, lr=args.lr,
                         momentum=0.9, nesterov=args.nesterov)

    args.epochs = math.ceil(args.total_steps / args.eval_step)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, args.warmup, args.total_steps)

    if args.use_ema:
        from models.ema import ModelEMA
        ema_model = ModelEMA(args, model, args.ema_decay)

    args.start_epoch = 0

    if args.resume:
        logger.info("==> Resuming from checkpoint..")
        assert os.path.isfile(
            args.resume), "Error: no checkpoint directory found!"
        args.out = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        best_acc_valid = checkpoint['best_acc_valid']
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        
        if args.use_ema:
            ema_model.ema.load_state_dict(checkpoint['ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])

    if args.amp:
        from apex import amp
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=args.opt_level)

    logger.info("***** Running training *****")
    logger.info(f"  Task = {args.dataset_name}@{args.num_labeled}")
    logger.info(f"  Num Epochs = {args.epochs}")
    logger.info(f"  Batch size per GPU = {args.batch_size}")
    logger.info(
        f"  Total train batch size = {args.batch_size}")
    logger.info(f"  Total optimization steps = {args.total_steps}\n\n")

    model.zero_grad()
    train(args, labeled_trainloader, unlabeled_trainloader, test_loader,val_loader,
          model, optimizer, ema_model, scheduler)

def train(args, labeled_trainloader, unlabeled_trainloader, test_loader, val_loader,
          model, optimizer, ema_model, scheduler):
    if args.amp:
        from apex import amp
    global best_acc, best_acc_valid
    test_accs = []
    valid_accs = []
    predictions = []
    end = time.time()

    labeled_iter = iter(labeled_trainloader)
    unlabeled_iter = iter(unlabeled_trainloader)

    model.train()
    for epoch in range(args.start_epoch, args.epochs):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        losses_x = AverageMeter()
        losses_u = AverageMeter()
        mask_probs = AverageMeter()
        logger.info('Epoch Number: {}\n'.format(epoch+1))
        if not args.no_progress:
            p_bar = tqdm(range(args.eval_step))
  
        for batch_idx in range(args.eval_step):
            try:
                inputs_x, targets_x = next(labeled_iter)
            except:
                labeled_iter = iter(labeled_trainloader)
                inputs_x, targets_x = next(labeled_iter)

            try:
                (inputs_u_w, inputs_u_s), targets_unlabeled = next(unlabeled_iter)
            except:
                unlabeled_iter = iter(unlabeled_trainloader)
                (inputs_u_w, inputs_u_s), targets_unlabeled = next(unlabeled_iter)

            data_time.update(time.time() - end)
            batch_size = inputs_x.shape[0]
            inputs = interleave(
                torch.cat((inputs_x, inputs_u_w, inputs_u_s)), 2*args.mu+1).to(args.device)
            targets_x = targets_x.to(args.device)
            targets_unlabeled = targets_unlabeled.to(args.device)
            logits = model(inputs)
            logits = de_interleave(logits, 2*args.mu+1)
            logits_x = logits[:batch_size]
            logits_u_w, logits_u_s = logits[batch_size:].chunk(2)
            
            del logits
            
            Lx = F.cross_entropy(logits_x, targets_x, reduction='mean')
            pseudo_label = torch.softmax(logits_u_w.detach(), dim=-1)
            max_probs, targets_u = torch.max(pseudo_label, dim=-1)
            
            #change mask for standard deviation based PL selection
            
            mask = max_probs.ge(args.threshold).float()

    ########################### Uncertainity Estimation ################################      
            # var = get_monte_carlo_predictions(inputs_u_w,targets_u,
            #                     10,
            #                     model,
            #                     get_num_classes(args.dataset_name),
            #                     len(inputs_u_w))

            # model.train()
            # # variance
            # var = var.to(device='cuda')
            # row_idxs = np.arange(var.shape[0])
            # col_idxs = targets_u
            # var_min = var[row_idxs, col_idxs]
            # mask_var = var_min.ge(0.2).float()
            # mask = mask_p*mask_var
            # del var, var_min
            
    ########################### Uncertainity Estimation ################################     
            Lu = (F.cross_entropy(logits_u_s, targets_u,
                                  reduction='none') * mask).mean()
            loss = Lx + args.lambda_u * Lu 
            
            
            if args.amp:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            losses.update(loss.item())
            losses_x.update(Lx.item())
            losses_u.update(Lu.item())
            optimizer.step()
            scheduler.step()
            if args.use_ema:
                ema_model.update(model)
            model.zero_grad()

            batch_time.update(time.time() - end)
            end = time.time()
            mask_probs.update(mask.mean().item())
            if not args.no_progress:
                p_bar.set_description("Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.4f}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. Loss_x: {loss_x:.4f}. Loss_u: {loss_u:.4f}. Mask: {mask:.2f}. ".format(
                    epoch=epoch + 1,
                    epochs=args.epochs,
                    batch=batch_idx + 1,
                    iter=args.eval_step,
                    lr=scheduler.get_last_lr()[0],
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,
                    loss_x=losses_x.avg,
                    loss_u=losses_u.avg,
                    mask=mask_probs.avg))
                p_bar.update()

        if not args.no_progress:
            p_bar.close()

        #Print out pseudo labels
        targets_un = targets_unlabeled[mask>0]
        targets_pl = targets_u[mask>0]
        correct = torch.sum(targets_un == targets_pl)
        assert len(targets_un) == len(targets_pl)
        total = len(targets_pl)
        if total > 0:
           pl_accuracy = correct / total
           #print("pseudo labels accuracy: ", pl_accuracy.item()*100)
           logger.info("Number of Pseudo Labels: {}\n".format(total))
           logger.info("Pseudo Labels Accuracy: {}".format(pl_accuracy*100))


        if args.use_ema:
            test_model = ema_model.ema
        else:
            test_model = model

        ###########################
        # if epoch == 0:
        #     test_model = ema_model.ema
        # else:
        #     test_model = create_model(args, args.dataset_name)
        #     # Load the state dicts of the three models
        #     best_model_state_dict = torch.load(args.out + '/model_best_valid.pth.tar')['state_dict']
        #     last_model_state_dict = torch.load(args.out + '/checkpoint.pth.tar')['state_dict']
        #     ema_model_state_dict = ema_model.ema.state_dict()
        #     # Combine the state dicts into a single dictionary
        #     combined_state_dict = {k: (best_model_state_dict[k] + last_model_state_dict[k] + ema_model_state_dict[k]) / 3
        #                            for k in best_model_state_dict.keys() & last_model_state_dict.keys() & ema_model_state_dict.keys()}


        #     # Load the combined state dict into the new model
        #     test_model.load_state_dict(combined_state_dict)
        #     test_model.cuda()
        #     print("Using Averaged Models")


            ##################################

            
        valid_loss, valid_acc = valid(args, val_loader, test_model, epoch)
        test_loss, test_acc = test(args, test_loader, test_model, epoch)

        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)

        is_best_valid = valid_acc > best_acc_valid
        best_acc_valid = max(valid_acc, best_acc_valid)

        model_to_save = model.module if hasattr(model, "module") else model

        model_to_save = model.module if hasattr(model, "module") else model
            
        if args.use_ema:
            ema_to_save = ema_model.ema.module if hasattr(
                ema_model.ema, "module") else ema_model.ema
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model_to_save.state_dict(),
                'ema_state_dict': ema_to_save.state_dict() if args.use_ema else None,
                'acc': test_acc,
                'best_acc': best_acc,
                'best_acc_valid': best_acc_valid,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, is_best,is_best_valid, args.out)
                        

            test_accs.append(test_acc)
            logger.info('Best top-1 test acc: {:.2f}'.format(best_acc))
            logger.info('Mean top-1 test acc: {:.2f}\n'.format(
                np.mean(test_accs[-20:])))

            valid_accs.append(valid_acc)
            logger.info('Best top-1  validation acc: {:.2f}'.format(best_acc_valid))
            logger.info('Mean top-1 validation acc: {:.2f}\n'.format(
                np.mean(valid_accs[-20:])))
            

            logger.info('--------------------------------NEXT EPOCH-------------------------------- \n')
            

def test(args, test_loader, model, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    if not args.no_progress:
        test_loader = tqdm(test_loader)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            data_time.update(time.time() - end)
            model.eval()

            inputs = inputs.to(args.device)
            targets = targets.to(args.device)
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)

            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.shape[0])
            top1.update(prec1.item(), inputs.shape[0])
            top5.update(prec5.item(), inputs.shape[0])
            batch_time.update(time.time() - end)
            end = time.time()
            if not args.no_progress:
                test_loader.set_description("Test Iter: {batch:4}/{iter:4}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. top1: {top1:.2f}. top5: {top5:.2f}. ".format(
                    batch=batch_idx + 1,
                    iter=len(test_loader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                ))
        if not args.no_progress:
            test_loader.close()

    logger.info("top-1 test acc: {:.2f}".format(top1.avg))
    logger.info("top-5 test acc: {:.2f}\n".format(top5.avg))
    return losses.avg, top1.avg


def valid(args, valid_loader, model, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    if not args.no_progress:
        valid_loader = tqdm(valid_loader)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valid_loader):
            data_time.update(time.time() - end)
            model.eval()

            inputs = inputs.to(args.device)
            targets = targets.to(args.device)
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)

            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.shape[0])
            top1.update(prec1.item(), inputs.shape[0])
            top5.update(prec5.item(), inputs.shape[0])
            batch_time.update(time.time() - end)
            end = time.time()
            if not args.no_progress:
                valid_loader.set_description("Valid Iter: {batch:4}/{iter:4}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. top1: {top1:.2f}. top5: {top5:.2f}. ".format(
                    batch=batch_idx + 1,
                    iter=len(valid_loader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                ))
        if not args.no_progress:
            valid_loader.close()

    logger.info("top-1 validation acc: {:.2f}".format(top1.avg))
    logger.info("top-5 validation acc: {:.2f}\n".format(top5.avg))
    return losses.avg, top1.avg



if __name__ == '__main__':
    main()