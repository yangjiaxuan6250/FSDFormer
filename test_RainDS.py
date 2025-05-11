from psnr_ssim import calculate_psnr, calculate_ssim
import os
from PIL import Image
import torchvision.transforms as transforms
def calculate_average(folder1, folder2):
    psnr_total = 0.0
    count = 0
    ssim_total = 0.0

    # 获取文件夹中相同名称的图片列表
    images_folder1 = os.listdir(folder1)
    images_folder2 = os.listdir(folder2)

    # 确保两个文件夹中都有相同名称的图片
    common_images = list(set(images_folder1).intersection(images_folder2))

    for img_name in common_images:
        img_path1 = os.path.join(folder1, img_name)
        img_path2 = os.path.join(folder2, img_name)
        img1 = Image.open(img_path1)
        img2 = Image.open(img_path2)
        transform = transforms.ToTensor()
        img1 = transform(img1)
        img2 = transform(img2)
        # 计算每对图片的 PSNR
        psnr = calculate_psnr(img1, img2)
        ssim = calculate_ssim(img1, img2)
        print(img_name)
        print('psnr:{}'.format(psnr))
        print('ssim:{}'.format(ssim))
        psnr_total += psnr
        ssim_total += ssim
        count += 1

    # 计算平均 PSNR
    average_psnr = psnr_total / count if count > 0 else 0.0
    average_ssim = ssim_total / count if count > 0 else 0.0
    return average_psnr, average_ssim

rain100l = ''
my = ''
if __name__ == '__main__':
    psnr, ssim = calculate_average(rain100l, my)
    print('psnr:{}'.format(psnr))
    print('ssim:{}'.format(ssim))

