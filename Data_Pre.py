import torch,visdom,time
import os,csv,random,glob
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader,Dataset

class Data(Dataset):
    def __init__(self,root,resize,model):
        super(Data, self).__init__()
        self.root=root
        self.resize=resize

        self.name2label={}   # {0,1,2……}
        for name in sorted(os.listdir(os.path.join(root))):
            if not os.path.isdir(os.path.join(root, name)):  # if path not exists that create
                continue
            self.name2label[name] = len(self.name2label.keys())

        self.images,self.labels=self.load_csv('images.csv')  # load data of label and images

        # dividing data
        if model=='train':
            self.images=self.images[:int(0.6*len(self.images))]
            self.labels=self.labels[:int(0.6*len(self.labels))]
        if model=='verify':
            self.images=self.images[int(0.6*len(self.images)):int(0.8*len(self.images))]
            self.labels=self.labels[int(0.6*len(self.labels)):int(0.8*len(self.labels))]
        else:
            self.images=self.images[int(0.8*len(self.images)):]
            self.labels=self.labels[int(0.8*len(self.labels)):]




    def load_csv(self,filename):
        if not os.path.exists(os.path.join(self.root, filename)):
            images = []
            for name in self.name2label.keys():
                images += glob.glob(os.path.join(self.root, name, '*.jpg'))
                # images += glob.glob(os.path.join(self.root, name, '*.png'))
                # images += glob.glob(os.path.join(self.root, name, '*.jpeg'))
            print(len(images), images)
            random.shuffle(images)
            with open(os.path.join(self.root, filename), mode='w', newline='') as f:
                writer = csv.writer(f)
                for img in images:
                    name = img.split(os.sep)[-2]
                    label = self.name2label[name]
                    writer.writerow([img, label])
                print('writen into csv file:', filename)
        images, labels = [], []
        with open(os.path.join(self.root, filename)) as f:
            reader = csv.reader(f)
            for row in reader:
                img, label = row
                label = int(label) # trans int for label
                images.append(img)
                labels.append(label)
        assert len(images) == len(labels)   # for len(images) and len(labels) is eq
        return images, labels

    def __len__(self):
        return len(self.images)

    def normalize(self, x_hat):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        # x_hat = (x-mean)/std
        # x = x_hat*std = mean
        # x: [c, h, w]
        # mean: [3] => [3, 1, 1]
        mean = torch.tensor(mean).unsqueeze(1).unsqueeze(1)
        std = torch.tensor(std).unsqueeze(1).unsqueeze(1)
        # print(mean.shape, std.shape)
        x = x_hat * std + mean
        return x
    def __getitem__(self, idx):
        img, label = self.images[idx], self.labels[idx]
        tf = transforms.Compose([
            lambda x:Image.open(x).convert('RGB'), # string path= > image data
            transforms.Resize((int(self.resize*1.25), int(self.resize*1.25))),
            transforms.RandomRotation(15),
            transforms.CenterCrop(self.resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        img = tf(img)
        label = torch.tensor(label)
        return img, label
def main():
    # show data
    viz = visdom.Visdom()
    db = Data('data', 64, 'train')

    x,y = next(iter(db))
    print('sample:', x.shape, y.shape, y)

    viz.image(db.normalize(x), win='sample_x', opts=dict(title='sample_x'))

    loader = DataLoader(db, batch_size=32, shuffle=True, num_workers=8)

    for x,y in loader:
        viz.images(db.normalize(x), nrow=8, win='batch', opts=dict(title='batch'))
        viz.text(str(y.numpy()), win='label', opts=dict(title='batch-y'))

        time.sleep(10)

if __name__ == '__main__':
    main()
