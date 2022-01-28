import argparse
import time
from models import *
from utils.datasets import *
from utils.utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=300, help='number of epochs')
parser.add_argument('--batch_size', type=int, default=4, help='size of each image batch')
parser.add_argument('--data_config_path', type=str, default='cfg/ICDAR2015.data', help='data config file path')
parser.add_argument('--cfg', type=str, default='cfg/yolov3.cfg', help='cfg file path')
parser.add_argument('--img_size', type=int, default=608, help='size of each image dimension')
parser.add_argument('--resume', type=bool, default=False, help='resume training from last.pt')
opt = parser.parse_args()
print(opt)

cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if cuda else 'cpu')

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
if cuda:
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = True

def main(opt):
    os.makedirs('weights', exist_ok=True)
    # Configure run
    data_config = parse_data_config(opt.data_config_path)
    num_classes = int(data_config['classes'])
    train_path = data_config['train']

    # Initialize model
    model = Darknet(opt.cfg, opt.img_size)

    # Get dataloader
    dataloader = load_images_and_labels(train_path, batch_size=opt.batch_size, img_size=opt.img_size, augment=False)
	
    # Reload saved optimizer state
    start_epoch = 0
    best_loss = float('inf')
    if opt.resume:
        checkpoint = torch.load('weights/latest.pt', map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        model.to(device).train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        start_epoch = checkpoint['epoch'] + 1
        if checkpoint['optimizer'] is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
            best_loss = checkpoint['best_loss']

        del checkpoint  # current, saved
    else:
        # Initialize model with darknet53 weights (optional) https://pjreddie.com/media/files/darknet53.conv.74
        load_weights(model, 'weights/darknet53.conv.74')
        model.to(device).train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    #model_info(model)
    t0, t1 = time.time(), time.time()
    mean_recall, mean_precision = 0, 0
    print('%11s' * 12 % ('Epoch', 'Batch', 'conf', 'cls', 'loss', 'P', 'R', 'nTargets', 'TP', 'FP', 'FN', 'time'))
    for epoch in range(opt.epochs):
        epoch += start_epoch

        # Update scheduler (manual) 
        if epoch < 30:
            lr = 1e-3
        elif epoch < 60:
            lr = 5e-4
        elif epoch < 90:
            lr = 1e-4
        else:
            lr = 5e-5

        for g in optimizer.param_groups:
            g['lr'] = lr

        ui = -1
        rloss = defaultdict(float)  # running loss
        metrics = torch.zeros(3, num_classes)
        optimizer.zero_grad()

        for i, (imgs, targets) in enumerate(dataloader):
            if sum([len(x) for x in targets]) < 1:  # if no targets continue
                continue
        
            # SGD burn-in
            if (epoch == 0) & (i <= 1000):
                lr = 1e-4 * (i / 1000) ** 4
                for g in optimizer.param_groups:
                    g['lr'] = lr

            # Compute loss, compute gradient, update parameters
            loss = model(imgs.to(device), targets, requestPrecision=True)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Compute running epoch-means of tracked metrics
            ui += 1
            metrics += model.losses['metrics']
            TP, FP, FN = metrics
            for key, val in model.losses.items():
                rloss[key] = (rloss[key] * ui + val) / (ui + 1)

            # Precision
            precision = TP / (TP + FP)
            k = (TP + FP) > 0
            if k.sum() > 0:
                mean_precision = precision[k].mean()

            # Recall
            recall = TP / (TP + FN)
            k = (TP + FN) > 0
            if k.sum() > 0:
                mean_recall = recall[k].mean()
			
            s = ('%11s%11s' + '%11.3g' * 10) % (
                    '%g/%g' % (epoch, opt.epochs - 1), '%g/%g' % (i, len(dataloader) - 1), rloss['conf'], rloss['cls'],
                    rloss['loss'], mean_precision, mean_recall, model.losses['nT'], model.losses['TP'],
                    model.losses['FP'], model.losses['FN'], time.time() - t1)
            t1 = time.time()
            print(s)
		

        # Write epoch results
        with open('results.txt', 'a') as file:
            file.write(s + '\n')

        # Update best loss
        loss_per_target = rloss['loss'] / rloss['nT']
        # Save latest checkpoint
        checkpoint = {'epoch': epoch,
					  'best_loss': best_loss,
					  'model': model.state_dict(),
					  'optimizer': optimizer.state_dict()}
        torch.save(checkpoint, 'weights/latest.pt')
        print("type(loss_per_target).....",type(loss_per_target))
        print("type(best_loss).....", type(best_loss))
        if loss_per_target < best_loss:
            best_loss = loss_per_target
            torch.save(checkpoint, 'weights/best.pt')

    # Save final model
    dt = time.time() - t0
    print('Finished %g epochs in %.2fs (%.2fs/epoch)' % (epoch, dt, dt / (epoch + 1)))

if __name__ == '__main__':
    torch.cuda.empty_cache()
    main(opt)
    torch.cuda.empty_cache()