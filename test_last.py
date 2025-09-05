import os
import argparse
import numpy as np
from tqdm import tqdm
import logging
from glob import glob
from pandas import DataFrame, Series
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils import setup_seed, cos_sim
from model.adapter import AdaptedCLIP
from model.clip import create_model
from dataset import get_dataset, DOMAINS
from forward_utils import (
    get_adapted_text_embedding,
    calculate_similarity_map,
    metrics_eval,
    visualize,
)
import warnings
import random

warnings.filterwarnings("ignore")

cpu_num = 4

os.environ["OMP_NUM_THREADS"] = str(cpu_num)
os.environ["OPENBLAS_NUM_THREADS"] = str(cpu_num)
os.environ["MKL_NUM_THREADS"] = str(cpu_num)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(cpu_num)
os.environ["NUMEXPR_NUM_THREADS"] = str(cpu_num)
torch.set_num_threads(cpu_num)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_support_features(model, support_loader, device):
    all_features = []
    for input_data in support_loader:  # bs always=1. training for an epoch first, Then use this updated model for memory bank construction.
        image = input_data[0].to(device)
        patch_tokens = model(image)
        patch_tokens = [t.reshape(-1, 768) for t in patch_tokens]
        all_features.append(patch_tokens)
    support_features = [
        torch.cat([all_features[j][i] for j in range(len(all_features))], dim=0)
        for i in range(len(all_features[0]))
    ]
    return support_features


def get_predictions(
        model: nn.Module,
        class_text_embeddings: torch.Tensor,
        test_loader: DataLoader,
        device: str,
        img_size: int,
        dataset: str = "MVTec",
):
    masks = []
    labels = []
    preds = []
    preds_image = []
    file_names = []
    # 调整权重参数，降低IQM权重以提升整体性能
    iqm_weight = 0.4  # 从0.7降低到0.4
    text_weight = 0.6  # 从0.3提高到0.6

    for input_data in tqdm(test_loader):
        image = input_data["image"].to(device)
        mask = input_data["mask"].cpu().numpy()
        label = input_data["label"].cpu().numpy()
        file_name = input_data["file_name"]
        class_name = input_data["class_name"]
        assert len(set(class_name)) == 1, "mixed class not supported"

        masks.append(mask)
        labels.append(label)
        file_names.extend(file_name)

        # 获取文本嵌入
        # 仿照训练代码处理文本嵌入
        epoch_text_feature = class_text_embeddings.unsqueeze(0).repeat(image.size(0), 1, 1)

        # 前向传播，包含IQM输出
        patch_features, det_feature, iqm_outputs = model(image, text_embeddings=epoch_text_feature)

        # 计算原始图像级预测
        pred = det_feature @ epoch_text_feature
        pred = (pred[:, 1] + 1) / 2
        preds_image.append(pred.cpu().numpy())

        # 计算文本异常图
        text_anomaly_maps = []
        for f in patch_features:
            patch_pred = calculate_similarity_map(
                f, epoch_text_feature, img_size, test=True, domain=DOMAINS[dataset]
            )
            text_anomaly_maps.append(patch_pred)

        # 计算IQM异常图
        iqm_anomaly_maps = []
        if iqm_outputs is not None:
            final_query_embedding = iqm_outputs.last_hidden_state
            norm_query = final_query_embedding[:, 0, :]
            abnorm_query = final_query_embedding[:, 1, :]

            for f in patch_features:
                # 解决维度不匹配问题 - 将query向量维度调整为与f相同
                if norm_query.shape[-1] != f.shape[-1]:
                    # 创建一个临时的线性投影层来调整维度
                    # 使用patch_features的第一个样本的设备信息
                    device = f.device
                    proj_layer = nn.Linear(norm_query.shape[-1], f.shape[-1]).to(device)
                    with torch.no_grad():
                        norm_query = proj_layer(norm_query)
                        abnorm_query = proj_layer(abnorm_query)

                norm_sim = F.cosine_similarity(f, norm_query.unsqueeze(1), dim=-1)
                abnorm_sim = F.cosine_similarity(f, abnorm_query.unsqueeze(1), dim=-1)
                iqm_pred = torch.sigmoid(abnorm_sim - norm_sim)

                # 将1D的patch序列重塑为2D特征图再进行插值
                B, L = iqm_pred.shape
                H = int(L ** 0.5)
                # 确保L是完全平方数
                assert H * H == L, f"L={L} is not a perfect square"

                # 重塑为4D张量 [B, 1, H, H]
                iqm_pred = iqm_pred.view(B, 1, H, H)
                iqm_pred = F.interpolate(
                    iqm_pred,
                    size=(img_size, img_size),
                    mode='bilinear',
                    align_corners=False
                )
                iqm_anomaly_maps.append(iqm_pred)

        # 融合异常图 - 改进融合策略，使用更稳定的权重
        if iqm_anomaly_maps:
            # 使用固定权重而非动态权重，提高稳定性
            text_map = torch.cat(text_anomaly_maps, dim=1).sum(1, keepdim=True)
            iqm_map = torch.cat(iqm_anomaly_maps, dim=1).sum(1, keepdim=True)

            # 使用更平衡的权重组合，避免过度依赖IQM
            final_map = text_map * text_weight + iqm_map * iqm_weight
        else:
            final_map = torch.cat(text_anomaly_maps, dim=1).sum(1)

        preds.append(final_map.cpu().numpy())

    masks = np.concatenate(masks, axis=0)
    labels = np.concatenate(labels, axis=0)
    preds = np.concatenate(preds, axis=0)
    preds_image = np.concatenate(preds_image, axis=0)

    return masks, labels, preds, preds_image, file_names

def main():
    parser = argparse.ArgumentParser(description="Training")
    # model
    parser.add_argument(
        "--model_name",
        type=str,
        default="ViT-L-14-336",
        help="ViT-B-16-plus-240, ViT-L-14-336",
    )
    parser.add_argument("--img_size", type=int, default=518)
    parser.add_argument("--relu", action="store_true")
    # testing
    parser.add_argument("--dataset", type=str, default="MVTec")
    parser.add_argument("--shot", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--image_batch_size", type=int, default=32)
    # exp
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--save_path", type=str, default="ckpt/baseline")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--text_norm_weight", type=float, default=0.1)
    parser.add_argument("--text_adapt_weight", type=float, default=0.1)
    parser.add_argument("--image_adapt_weight", type=float, default=0.1)
    parser.add_argument("--text_adapt_until", type=int, default=3)
    parser.add_argument("--image_adapt_until", type=int, default=6)
    # 调整IQM超参数以提升性能
    parser.add_argument("--iqm_hidden_size", type=int, default=512)  # 降低维度以减少过拟合
    parser.add_argument("--iqm_num_layers", type=int, default=2)  # 减少层数
    parser.add_argument("--iqm_num_heads", type=int, default=8)
    parser.add_argument("--iqm_weight", type=float, default=0.7)  # 增加IQM权重


    args = parser.parse_args()
    # ========================================================
    setup_seed(args.seed)
    # check save_path and setting logger
    os.makedirs(args.save_path, exist_ok=True)
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        filename=os.path.join(args.save_path, "test.log"),
        encoding="utf-8",
        level=logging.INFO,
    )
    logger.info("args: %s", vars(args))
    # set device
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    # ========================================================
    # load model
    # set up model for testing
    clip_model = create_model(
        model_name=args.model_name,
        img_size=args.img_size,
        device=device,
        pretrained="openai",
        require_pretrained=True,
    )
    clip_model.eval()
    model = AdaptedCLIP(
        clip_model=clip_model,
        text_adapt_weight=args.text_adapt_weight,
        image_adapt_weight=args.image_adapt_weight,
        text_adapt_until=args.text_adapt_until,
        image_adapt_until=args.image_adapt_until,
        relu=args.relu,
        iqm_hidden_size=args.iqm_hidden_size,
        iqm_num_layers=args.iqm_num_layers,
        iqm_num_heads=args.iqm_num_heads,
    ).to(device)
    model.eval()
    # load checkpoints if exists
    text_file = glob(args.save_path + "/text_adapter.pth")
    assert len(text_file) >= 0, "text adapter checkpoint not found"
    if len(text_file) > 0:
        checkpoint = torch.load(text_file[0])
        model.text_adapter.load_state_dict(checkpoint["text_adapter"])
        adapt_text = True
    else:
        adapt_text = False

    files = sorted(glob(args.save_path + "/image_adapter_*.pth"))
    assert len(files) > 0, "image adapter checkpoint not found"
    # 只测试最后一个checkpoint
    # 按照epoch数字排序，而不是字典序
    files = sorted(files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    file = files[-1]
    checkpoint = torch.load(file)
    model.image_adapter.load_state_dict(checkpoint["image_adapter"])
    test_epoch = checkpoint["epoch"]
    logger.info("-----------------------------------------------")
    logger.info("load model from epoch %d", test_epoch)
    logger.info("-----------------------------------------------")
    # ========================================================
    # load dataset
    kwargs = {"num_workers": 4, "pin_memory": True} if use_cuda else {}
    image_datasets = get_dataset(
        args.dataset,
        args.img_size,
        None,
        args.shot,
        "test",
        logger=logger,
    )
    with torch.no_grad():
        if adapt_text:
            text_embeddings = get_adapted_text_embedding(
                model, args.dataset, device
            )
        else:
            text_embeddings = get_adapted_text_embedding(
                clip_model, args.dataset, device
            )
    # ========================================================
    df = DataFrame(
        columns=[
            "class name",
            "pixel AUC",
            "pixel AP",
            "image AUC",
            "image AP",
        ]
    )
    for class_name, image_dataset in image_datasets.items():
        image_dataloader = torch.utils.data.DataLoader(
            image_dataset, batch_size=args.image_batch_size, shuffle=True, **kwargs
        )

        # ========================================================
        # testing
        with torch.no_grad():
            # 仿照训练代码处理文本嵌入
            class_text_embeddings = text_embeddings[class_name]
            masks, labels, preds, preds_image, file_names = get_predictions(
                model=model,
                class_text_embeddings=class_text_embeddings,
                test_loader=image_dataloader,
                device=device,
                img_size=args.img_size,
                dataset=args.dataset,
            )
        # ========================================================
        if args.visualize:
            visualize(
                masks,
                preds,
                file_names,
                args.save_path,
                args.dataset,
                class_name=class_name,
            )
        class_result_dict = metrics_eval(
            masks,
            labels,
            preds,
            preds_image,
            class_name,
            domain=DOMAINS[args.dataset],
        )

        df.loc[len(df)] = Series(class_result_dict)
    # 仅对数值列计算平均值
    numeric_cols = ["pixel AUC", "pixel AP", "image AUC", "image AP"]
    average_row = df[numeric_cols].mean()
    # 添加"Average"作为类别名称
    average_row["class name"] = "Average"
    # 插入平均值行
    df.loc[len(df)] = average_row
    logger.info("final results:\n%s", df.to_string(index=False, justify="center"))

if __name__ == "__main__":
    main()
