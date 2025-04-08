import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import cv2
from PIL import Image, ImageFont, ImageDraw, ImageFilter
from .HieroglyphCharacterGenerator import HieroglyphCharacterGenerator
from .HieroglyphAugmentator import HieroglyphAugmentator
from .HieroglyphDataset import HieroglyphDataset
from .CustomMorphOps import resize_to_square

CHUNK_SIZE = 10 #Plot of (20,20) subplots

paths = [ 
        "../../files/fonts/Noto_Sans_Egyptian_Hieroglyphs/NotoSansEgyptianHieroglyphs-Regular.ttf",
        "../../files/fonts/NewGardiner/NewGardinerBMP.ttf",
        "../../files/fonts/JSeshFont/JSeshFont.ttf",
            ]   
ranges = [ 
    (0x00013000, 0x0001342E),
    (0x0000E000, 0x0000E42E),
    (0x00013000, 0x0001342E),
        ]


def plot_predictions_table(predictions, targets, epoch):

    print(f"len(predictions):{len(predictions)}")
    print(f"len(targets):{len(targets)}")

    # Convert to regular Python lists if they're numpy arrays
    if isinstance(predictions, np.ndarray):
        predictions = predictions.tolist()
    if isinstance(targets, np.ndarray):
        targets = targets.tolist()
        
    # Create figure and axis
    plt.figure(figsize=(2, 200))
    ax = plt.subplot(111)
    ax.axis('off')
    
    # Limit to first 50 predictions to keep table manageable
    max_display = len(predictions)
    
    # Prepare data for table
    table_data = []
    for i in range(max_display):
        table_data.append([targets[i], predictions[i]])
    
    # Create table
    column_labels = ['Target', 'Prediction']
    table = ax.table(
        cellText=table_data,
        colLabels=column_labels,
        loc='center',
        cellLoc='center'
    )

    for row_idx in range(len(table_data)):
        if predictions[row_idx] == targets[row_idx]:
            color = "#66ff66" #green
        else:
            color = "#ff6666" #red
        for col_idx in range(len(column_labels)):
            table[(row_idx + 1, col_idx)].set_facecolor(color)
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    
    # Add headers
    plt.title(f'Predictions vs Targets - Epoch {epoch}', y=1.08)
    
    # Save figure
    plt.subplots_adjust(left=0.1, right=0.9, top=0.95, bottom=0.05)
    plt.savefig(f'../results/predictions_table_epoch_{epoch}', bbox_inches='tight', dpi=150)
    plt.close()

def tagPredictOnImage(img, target, predict, correct):
    img_copy = Image.fromarray(img)
    draw = ImageDraw.Draw(img_copy)
    font = ImageFont.truetype("/home/dcorr/.local/share/fonts/1Nunito-VariableFont_wght.ttf",24)
    color = (0,255,0) if correct else (255,0,0)
    draw.text((0,0), str(predict), color, font=font)
    return np.array(img_copy)
    
def tensorToRGBImage(img_tensor):
    if isinstance(img_tensor, torch.Tensor):
        img_tensor = img_tensor.permute(1,2,0).cpu().numpy()
    if img_tensor.shape[2] == 1:
        img_tensor = img_tensor[:,:,0]
    # Convertir a uint8
    bgr = cv2.cvtColor(img_tensor, cv2.COLOR_GRAY2BGR)
    return cv2.cvtColor(((bgr * 255).astype(np.uint8)), cv2.COLOR_BGR2RGB)

def concat2img(img_0, img_1):
    shape0 = img_0.shape
    shape1 = img_1.shape
    if shape0 == shape1:
        black_canvas = np.zeros((shape0[0], shape0[0] * 2), dtype=np.uint8)
        black_canvas[:shape0[0], :shape0[0]] = img_0[:,:,0]
        black_canvas[:shape0[0], shape0[0] : shape0[0] * 2] = img_1[:,:,0]
        bgr = cv2.cvtColor(black_canvas, cv2.COLOR_GRAY2BGR)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    
def compareDatasetPredicts(test_dataset: HieroglyphDataset,
                           targets: list,
                           predicts: list,
                           experiment: int = 0):
    print(targets[:5])
    print(predicts[:5])
    if len(targets) == len(predicts):
        common_len = len(targets)
        n_chunks = common_len // (CHUNK_SIZE ** 2)
        if common_len % (CHUNK_SIZE ** 2) > 0: n_chunks+=1
        for i in range(n_chunks):
            fig, axes = plt.subplots(CHUNK_SIZE, CHUNK_SIZE, figsize=(10,10))
            for j in range(CHUNK_SIZE):
                for k in range(CHUNK_SIZE):
                    target = i * (CHUNK_SIZE ** 2) + j * CHUNK_SIZE + k
                    if target < common_len:
                        predict = predicts[target]
                        target_tensor_img, target_label = test_dataset[target]
                        predict_tensor_img, predict_label = test_dataset[predict]
                        axes[j,k].imshow(
                            concat2img(
                                tensorToRGBImage(target_tensor_img),
                                tensorToRGBImage(predict_tensor_img)
                                )
                        )
                        color = "#66ff66" if target == predict else "#ff6666"
                        axes[j,k].set_title(f"{target} -> {predict}", color=color, fontsize=8, fontweight='bold')
                    axes[j,k].axis('off')
            plt.tight_layout()
            fig.savefig(f"{experiment}_{i}_{j}_{k}.png")

    


def main():
    all_generators = []
    for path,hex_range in zip(paths,ranges):
        all_generators.append(HieroglyphCharacterGenerator(path, hex_range[0], hex_range[1], font_size=100))
        augmentator = HieroglyphAugmentator(all_generators, mask=(3,3), fill=False)
    generator_len = all_generators[0].getFontLength()
    dataset = HieroglyphDataset(generator_len, augmentator)
    random.seed(33)
    targets = [x for x in range(1071)]
    predicts = [x if random.randint(0, 2) == 1 else random.randint(0, 1071) for x in range(1071)]

    #plot_predictions_table(predicts, targets, 1)
    (img_tensor0, label0) = dataset[0]
    print(type(img_tensor0))
    print(img_tensor0.shape)
    img0 = tensorToRGBImage(img_tensor0)
    print(img0.shape)
    plt.imsave("./true_predict.png", tagPredictOnImage(img0, label0, label0, True))
    plt.imsave("./false_predict.png", tagPredictOnImage(img0, label0, 1064, False))
    concatimg = concat2img(img0, img0)
    print(concatimg.shape)
    plt.imsave("./concat2img.png", concatimg)
    compareDatasetPredicts(dataset, targets, predicts)

if __name__ == "__main__":
    main()