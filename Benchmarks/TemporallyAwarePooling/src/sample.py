from dataset import SoccerNetCaptions, collate_fn_padd
from model import Video2Caption
from SoccerNet.Evaluation.utils import AverageMeter
import torch
import numpy as np
import time
from tqdm import tqdm
from loguru import logger as logging
from datetime import datetime

logging.add(
    "logs/sample.log", format="{time} {level} {message}", level="INFO", rotation="10 MB"
)


def validate_captioning(dataloader, model, model_name):
    batch_time = AverageMeter()
    data_time = AverageMeter()

    model.eval()

    end = time.time()
    all_labels = []
    all_outputs = []

    try:
        with tqdm(dataloader) as t:
            for (feats, caption), lengths, mask, caption_or, cap_id in t:
                # measure data loading time
                data_time.update(time.time() - end)
                feats = feats.cuda()
                # compute output string
                output = [
                    dataloader.dataset.detokenize(
                        list(model.sample(feats[idx]).detach().cpu())
                    )
                    for idx in range(feats.shape[0])
                ]

                all_outputs.extend(output)
                all_labels.extend(caption_or)
                logging.info(f"Output: {output}, Label: {caption_or}, ID: {cap_id}")

                batch_time.update(time.time() - end)
                end = time.time()

                desc = "Test (cap): "
                desc += f"Time {batch_time.avg:.3f}s "
                desc += f"(it:{batch_time.val:.3f}s) "
                desc += f"Data:{data_time.avg:.3f}s "
                desc += f"(it:{data_time.val:.3f}s) "
                t.set_description(desc)
    except KeyboardInterrupt:
        t.close()
        logging.info("Exiting from training early")

    with open(f"logs/{datetime.now().strftime()}_{model_name}.tsv", "w") as f:
        for output, label in zip(all_outputs, all_labels):
            f.write(f"{output}\t{label}\n")


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    vlad_k = 64
    num_layers = 4
    framerate = 1
    window_size = 45
    root = "/raid_elmo/home/lr/moriy/SoccerNet/"
    dataset_Test = SoccerNetCaptions(
        path=root,
        features="baidu_soccer_embeddings.npy",
        split=["test"],
        version=2,
        framerate=framerate,
        window_size=window_size,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset_Test,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        collate_fn=collate_fn_padd,
    )

    # model_name = "baidu-NetVlad-caption-20230713"
    # weights = "models/baidu-NetVlad-caption-20230713"
    # model_name = "20230606_new_model-caption"
    # weights = "models/20230606_new_model-caption"
    model_name = "sn_benchmark_new_model"
    weights = f"models/{model_name}/caption/model.pth.tar"

    feature_dim = dataset_Test[0][0].shape[-1]
    vocab_size = dataset_Test.vocab_size
    model = Video2Caption(
        vocab_size,
        pool="NetVLAD",
        input_size=feature_dim,
        framerate=framerate,
        window_size=window_size,
        vlad_k=vlad_k,
        num_layers=num_layers,
        weights_encoder=weights,
    )
    model.cuda()
    score = validate_captioning(test_loader, model, model_name)
    print(f"Score: {score}")
