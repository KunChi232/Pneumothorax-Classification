from VGG_16 import VGG
import os
def main():
    batch_size = 15
    num_epoch = 30
    os.environ['CUDA_VISIBLE_DEVICES'] = ''    
    VGG_16 = VGG()
    g = VGG_16.build_network()
    VGG_16.train_network(g, batch_size, num_epoch)

if __name__ == '__main__':
    main()