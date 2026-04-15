import os
from PIL import Image
from torchvision.datasets import VisionDataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

class MyImageDataset(VisionDataset):
    def __init__(self, root, transform=None):
        # 初始化父类，把 root 和 transform 传进去
        super().__init__(root, transform=transform)
        
        self.image_paths = [] # 用来存所有图片的绝对路径
        self.labels = []      # 用来存对应的数字标签
        
        # 定义一个字典，把字符串类别变成机器认识的数字
        self.class_to_idx = {"cats": 0, "dogs": 1}
        
        # 【核心动作】：遍历硬盘，建立索引目录（绝对不在这里读图！）
        for class_name in self.class_to_idx.keys():
            class_dir = os.path.join(root, class_name)
            if not os.path.isdir(class_dir):
                continue
                
            # 拿到这个类别文件夹下的所有图片文件名
            for file_name in os.listdir(class_dir):
                if file_name.endswith('.jpg') or file_name.endswith('.png'):
                    # 把绝对路径拼好，存进列表
                    img_path = os.path.join(class_dir, file_name)
                    self.image_paths.append(img_path)
                    # 把对应的数字标签存进列表
                    self.labels.append(self.class_to_idx[class_name])

    def __len__(self):
        # 告诉 DataLoader，我们总共有多少张图
        return len(self.image_paths)

    def __getitem__(self, index):
        # 【核心动作】：根据索引拿数据（这才是真正的懒加载读取）
        
        # 1. 拿到这张图的路径和标签
        img_path = self.image_paths[index]
        label = self.labels[index]
        
        # 2. 从硬盘读取这张图 (一定要转成 RGB，防止有灰度图或者透明通道干扰)
        image = Image.open(img_path).convert("RGB")
        
        # 3. 如果定义了预处理，就在这里执行（比如转 Tensor，缩放）
        if self.transform is not None:
            image = self.transform(image)
            
        # 4. 完美打包返回
        return image, label

# ================= 测试运行 =================

# 1. 定义预处理流程
my_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 缩放到标准大小
    transforms.ToTensor(),          # 变成 [C, H, W] 的张量
])

# 2. 实例化数据集
# 此时，__init__ 被执行，建好了目录索引，但硬盘里的图片还没有被真正读取
my_dataset = MyImageDataset(root='./my_data', transform=my_transform)

print(f"成功加载数据集，共有 {len(my_dataset)} 张图片。")

# 3. 挂载到 DataLoader
# 这里自动帮你做了上一问提到的“组装大集装箱（Batch）”的工作
my_dataloader = DataLoader(dataset=my_dataset, batch_size=32, shuffle=True)

# 4. 模拟训练循环
for step, (batch_images, batch_labels) in enumerate(my_dataloader):
    # 此时，__getitem__ 才会被疯狂调用 32 次去读图，然后打包成这一个 Batch 返回
    print(f"Step {step}:")
    print(f"  输入的图片张量形状: {batch_images.shape}") # 应该是 [32, 3, 224, 224]
    print(f"  输入的标签张量形状: {batch_labels.shape}") # 应该是 [32]
    break # 测试一下就停