import argparse
import time
import torch
import torch.nn as nn
import os
import sys
from collections import Counter
from pathlib import Path


from qfnn.qf_fb.c_input import load_data,modify_target
from qfnn.qf_net.utils import binarize,save_checkpoint
from qfnn.qf_fb.c_qf_mixer import Net


# from qiskit_simulator_wbn import run_simulator

import logging
logging.basicConfig(stream=sys.stdout,
                    level=logging.WARNING,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='QuantumFlow Classification Training')

    # ML related
    parser.add_argument('--device', default='cpu', help='device')
    parser.add_argument('-c','--interest_class',default="3,6",help="investigate classes",)
    parser.add_argument('-s','--img_size', default="4", help="image size 4: 4*4", )
    parser.add_argument('-dp', '--datapath', default='/home/hzr/Software/quantum/qc_mnist/pytorch/data', help='dataset')
    parser.add_argument('-ppd', "--preprocessdata", help="Using the preprocessed data", action="store_true", )
    parser.add_argument('-j','--num_workers', default="0", help="worker to load data", )
    parser.add_argument('-tb','--batch_size', default="64", help="training batch size", )
    parser.add_argument('-ib','--inference_batch_size', default="32", help="inference batch size", )
    parser.add_argument('-nn','--neural_in_layers', default="u:4,p:2", help="QNN structrue :<layer1 name: output_number;layer2 name:output_number...>", )

    parser.add_argument('-l','--init_lr', default="0.001", help="PNN learning rate", )
    parser.add_argument('-m','--milestones', default="5, 7, 9", help="Training milestone", )
    parser.add_argument('-e','--max_epoch', default="10", help="Training epoch", )
    parser.add_argument('-r','--resume_path', default='', help='resume from checkpoint')
    parser.add_argument('-t',"--test_only", help="Only Test without Training", action="store_true", )
    parser.add_argument('-bin', "--binary",default=False , help="binary activation", action="store_true", )

    # QC related
    #parser.add_argument('-wn', "--with_norm", default=False ,help="Using Batchnorm", action="store_true", )
    parser.add_argument('-qa',"--given_ang", default="1 -1 1 -1, -1 -1",  help="ang amplify, the same size with --neural_in_layers",)
    parser.add_argument('-qt',"--train_ang", help="train anglee", action="store_true", )


    # File
    parser.add_argument('-chk',"--save_chkp",default=True , help="Save checkpoints", action="store_true", )
    parser.add_argument('-chkname', '--chk_name', default='', help='folder name for chkpoint')
    parser.add_argument('-chkpath', '--chk_path', default='.', help='Root folder for chkpoint')

    args = parser.parse_args()
    return args


def train(epoch,interest_num,criterion,train_loader):
    model.train()
    correct = 0
    epoch_loss = []
    for batch_idx, (data, target) in enumerate(train_loader):
        # data = ToQuantumData_Batch()(data)
        target, new_target = modify_target(target,interest_num)

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data, True)
        # print("output",output)

        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        loss = criterion(output, target)
        epoch_loss.append(loss.item())
        loss.backward()

        optimizer.step()

        if batch_idx % 25 == 0:
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {}/{} ({:.2f}%)'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss, correct, (batch_idx + 1) * len(data),
                       100. * float(correct) / float(((batch_idx + 1) * len(data)))))
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {}/{} ({:.2f}%)'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss, correct, (batch_idx + 1) * len(data),
                       100. * float(correct) / float(((batch_idx + 1) * len(data)))))
    print("-" * 20, "training done, loss", "-" * 20)
    logger.info("Training Set: Average loss: {}".format(round(sum(epoch_loss) / len(epoch_loss), 6)))


accur = []

def test(interest_num,criterion,test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        # data = ToQuantumData_Batch()(data)
        target, new_target = modify_target(target,interest_num)
        data, target = data.to(device), target.to(device)
        output = model(data, False)
        test_loss += criterion(output, target)  # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum() # get the right number

    a = 100. * correct / len(test_loader.dataset)
    accur.append(a)
    test_loss /= len(test_loader.dataset)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * float(correct) / float(len(test_loader.dataset))))

    return float(correct) / len(test_loader.dataset)





if __name__ == "__main__":

    print("=" * 100)
    print("Training procedure for Quantum Computer:")
    print("\tStart at:", time.strftime("%m/%d/%Y %H:%M:%S"))
    print("\tProblems and issues, please contact Dr. Weiwen Jiang (wjiang2@nd.edu)")
    print("\tEnjoy and Good Luck!")
    print("=" * 100)
    print()

    args = parse_args()

    datapath = args.datapath
    chkpath = args.chk_path
    isppd = args.preprocessdata
    device = args.device
    interest_class = [int(x.strip()) for x in args.interest_class.split(",")]
    img_size = int(args.img_size)
    num_workers = int(args.num_workers)
    batch_size = int(args.batch_size)
    inference_batch_size = int(args.inference_batch_size)
    given_ang = [[int(y) for y in x.strip().split(" ")] for x in args.given_ang.split(",")]
    train_ang = args.train_ang
    layers =[]

    for item1 in args.neural_in_layers.split(","):
        x= item1.split(":")
        layer =[]
        layer.append(x[0].strip())
        layer.append(int(x[1].strip()))
        layers.append(layer)
    
    print("layers:",layers)

    init_lr = float(args.init_lr)
    milestones = [int(x.strip()) for x in args.milestones.split(",")]
    max_epoch = int(args.max_epoch)
    resume_path = args.resume_path
    training = not(args.test_only)
    binary = args.binary
    # with_norm = args.with_norm

    save_chkp = args.save_chkp
    if save_chkp:
        if args.chk_name!="":
            save_path = chkpath + "/model/" + args.chk_name
        else:
            save_path = chkpath + "/model/" + os.path.basename(sys.argv[0]) + "_" + time.strftime("%Y_%m_%d-%H_%M_%S")
        Path(save_path).mkdir(parents=True, exist_ok=True)

        logger.info("Checkpoint path: {}".format(save_path))
        print(("Checkpoint path: {}".format(save_path)))

    if save_chkp:
        fh = open(save_path+"/config","w")
        print("=" * 21, "Your setting is listed as follows", "=" * 22, file=fh)
        print("\t{:<25} {:<15}".format('Attribute', 'Input'), file=fh)
        for k, v in vars(args).items():
            print("\t{:<25} {:<15}".format(k, v), file=fh)
        print("=" * 22, "Exploration will start, have fun", "=" * 22, file=fh)
        print("=" * 78, file=fh)

    print("=" * 21, "Your setting is listed as follows", "=" * 22)
    print("\t{:<25} {:<15}".format('Attribute', 'Input'))
    for k,v in vars(args).items():
        print("\t{:<25} {:<15}".format(k, v))
    print("=" * 22, "Exploration will start, have fun", "=" * 22)
    print("=" * 78)


    # Schedule train and test

    train_loader, test_loader = load_data(interest_class,datapath,isppd,img_size,batch_size,inference_batch_size,num_workers)
    criterion = nn.CrossEntropyLoss()
    model = Net(img_size,layers,training,binary,given_ang,train_ang)
    model = model.to(device)

    print(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=init_lr)
    
    for item in model.parameters():
        print("model.parameter:",item)
  
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1)
    # print(resume_path,os.path.isfile(resume_path))

    if os.path.isfile(resume_path):
        print("=> loading checkpoint from '{}'<=".format(resume_path))
        checkpoint = torch.load(resume_path, map_location=device)
        epoch_init, acc = checkpoint["epoch"], checkpoint["acc"]
        model.load_state_dict(checkpoint["state_dict"])


        scheduler.load_state_dict(checkpoint["scheduler"])
        scheduler.milestones = Counter(milestones)
        optimizer.load_state_dict(checkpoint["optimizer"])
    else:
        epoch_init, acc = 0, 0



    if training:
        for epoch in range(epoch_init, max_epoch + 1):
            print("=" * 20, epoch, "epoch", "=" * 20)
            print("Epoch Start at:", time.strftime("%m/%d/%Y %H:%M:%S"))

            print("-" * 20, "learning rates", "-" * 20)
            for param_group in optimizer.param_groups:
                print(param_group['lr'], end=",")
            print()

            print("-" * 20, "training", "-" * 20)
            print("Trainign Start at:", time.strftime("%m/%d/%Y %H:%M:%S"))
            train(epoch,interest_class,criterion,train_loader)
            print("Trainign End at:", time.strftime("%m/%d/%Y %H:%M:%S"))
            print("-" * 60)


            print()

            print("-" * 20, "testing", "-" * 20)
            print("Testing Start at:", time.strftime("%m/%d/%Y %H:%M:%S"))
            cur_acc = test(interest_class,criterion,test_loader)
            print("Testing End at:", time.strftime("%m/%d/%Y %H:%M:%S"))
            print("-" * 60)
            print()




            scheduler.step()

            is_best = False
            if cur_acc > acc:
                is_best = True
                acc = cur_acc

            print("Best accuracy: {}; Current accuracy {}. Checkpointing".format(acc, cur_acc))


            if save_chkp:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'acc': acc,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                }, is_best, save_path, 'checkpoint_{}_{}.pth.tar'.format(epoch, round(cur_acc, 4)))
            print("Epoch End at:", time.strftime("%m/%d/%Y %H:%M:%S"))
            print("=" * 60)
            print()
    else:
        print("=" * 20, max_epoch, "Testing", "=" * 20)
        print("=" * 100)
        for name, para in model.named_parameters():
            if "fc" in name:
                print(name,binarize(para))
            else:
                print(name, para)
        print("="*100)

        q_start = time.time()
        test(interest_class, criterion, test_loader)
        q_end = time.time()

        print(q_start-q_end)




