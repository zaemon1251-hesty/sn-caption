from SoccerNet.utils import getListGames
from SoccerNet.Evaluation.utils import LoadJsonFromZip, getMetaDataTask
from SoccerNet.Evaluation.ActionSpotting import average_mAP
import os
from tqdm import tqdm
import zipfile
import json
import numpy as np
import glob
import argparse


def evaluate_commentary_spotting(
    labels_list: np.ndarray,
    predictions_list: np.ndarray,
    metric="loose",
):
    """
    evaluate() を参考にして、コメントのスポッティングの評価を行う
    Params:
    - labels_list: list of labels(numpy array)
    - predictions_list: list of predictions(numpy array)
    - metric: metric to evaluate from ["loose", "tight", "medium"]
    Return:
    - details mAP
    """
    framerate = 1

    targets_numpy = labels_list
    detections_numpy = predictions_list
    closests_numpy = list()

    for labels in tqdm(labels_list):
        closest_numpy = get_closest_label_numpy(labels)
        closests_numpy.append(closest_numpy)

    if metric == "loose":
        deltas = np.arange(12) * 5 + 5
    elif metric == "tight":
        deltas = np.arange(5) * 1 + 1
    elif metric == "medium":
        deltas = np.array([30])

    # Compute the performances
    (
        a_mAP,
        a_mAP_per_class,
        _,
        _,
        _,
        _,
    ) = average_mAP(
        targets_numpy, detections_numpy, closests_numpy, framerate, deltas=deltas
    )

    results = {
        "a_mAP": a_mAP,
        "a_mAP_per_class": a_mAP_per_class,
        "a_mAP_visible": None,
        "a_mAP_per_class_visible": None,
        "a_mAP_unshown": None,
        "a_mAP_per_class_unshown": None,
    }
    return results


def evaluate(
    SoccerNet_path,
    Predictions_path,
    prediction_file="results_spotting.json",
    split="test",
    version=2,
    framerate=2,
    metric="loose",
):
    """
    evaluate the prediction with respect to some ground truth
    Params:
    - SoccerNet_path: path for labels (folder or zipped file)
    - Predictions_path: path for predictions (folder or zipped file)
    - prediction_file: name of the predicted files - if set to None, try to infer it
    - split: split to evaluate from ["test", "challenge"]
    - frame_rate: frame rate to evalaute from [2]
    Return:
    - details mAP
    """
    list_games = getListGames(split=split, task="caption")
    targets_numpy = list()
    detections_numpy = list()
    closests_numpy = list()

    label_files, num_classes, _, _ = getMetaDataTask("caption", "SoccerNet", version)

    for game in tqdm(list_games):
        if zipfile.is_zipfile(SoccerNet_path):
            labels = LoadJsonFromZip(SoccerNet_path, os.path.join(game, label_files))
        else:
            labels = json.load(open(os.path.join(SoccerNet_path, game, label_files)))
        # convert labels to vector

        # TODO label_half_1, label_half_2 を引数として渡せるようにする
        label_half_1, label_half_2 = label2vector(
            labels, num_classes=num_classes, version=version, framerate=framerate
        )

        # infer name of the prediction_file
        # TODO 消す
        if prediction_file is None:
            if zipfile.is_zipfile(Predictions_path):
                with zipfile.ZipFile(Predictions_path, "r") as z:
                    for filename in z.namelist():
                        if filename.endswith(".json"):
                            prediction_file = os.path.basename(filename)
                            break
            else:
                for filename in glob.glob(
                    os.path.join(Predictions_path, "*/*/*/*.json")
                ):
                    prediction_file = os.path.basename(filename)
                    break

        # Load predictions
        if zipfile.is_zipfile(Predictions_path):
            predictions = LoadJsonFromZip(
                Predictions_path, os.path.join(game, prediction_file)
            )
        else:
            predictions = json.load(
                open(os.path.join(Predictions_path, game, prediction_file))
            )
        # convert predictions to vector
        # TODO predictions_half_1, predictions_half_2 を引数として渡せるようにする
        predictions_half_1, predictions_half_2 = predictions2vector(
            predictions, num_classes=num_classes, version=version, framerate=framerate
        )

        targets_numpy.append(label_half_1)
        targets_numpy.append(label_half_2)
        detections_numpy.append(predictions_half_1)
        detections_numpy.append(predictions_half_2)

        closests_numpy.append(get_closest_label_numpy(label_half_1))
        closests_numpy.append(get_closest_label_numpy(label_half_2))

    if metric == "loose":
        deltas = np.arange(12) * 5 + 5
    elif metric == "tight":
        deltas = np.arange(5) * 1 + 1
    elif metric == "medium":
        deltas = np.array([30])
    # Compute the performances
    (
        a_mAP,
        a_mAP_per_class,
        a_mAP_visible,
        a_mAP_per_class_visible,
        a_mAP_unshown,
        a_mAP_per_class_unshown,
    ) = average_mAP(
        targets_numpy, detections_numpy, closests_numpy, framerate, deltas=deltas
    )

    results = {
        "a_mAP": a_mAP,
        "a_mAP_per_class": a_mAP_per_class,
        "a_mAP_visible": a_mAP_visible if version == 2 else None,
        "a_mAP_per_class_visible": a_mAP_per_class_visible if version == 2 else None,
        "a_mAP_unshown": a_mAP_unshown if version == 2 else None,
        "a_mAP_per_class_unshown": a_mAP_per_class_unshown if version == 2 else None,
    }
    return results


def get_closest_label_numpy(label_numpy: np.ndarray) -> np.ndarray:
    # Get the closest action index
    closest_numpy = np.zeros(label_numpy.shape) - 1

    for c in np.arange(label_numpy.shape[-1]):  # c is class index
        # np.where(label_half_1[:, c] != 0) = class c を割り当てる確率が0ではないフレームのインデックス配列(x軸のindexを格納した配列と軸のindexを格納した配列のタプル)
        # np.where(label_half_1[:, c] != 0)[0].tolist() = 上の配列の最初の要素(x軸のindex)をリスト化する
        indexes = np.where(label_numpy[:, c] != 0)[0].tolist()

        if len(indexes) == 0:
            continue

        # あり得ないくらい小さいindexを一番前に入れることで、label_halfの最初の要素に対しても処理を行えるようにする
        indexes.insert(0, -indexes[0])

        # あり得ないくらい大きいindexを一番後ろに追加することで、label_halfの最後の要素に対しても処理を行えるようにする
        indexes.append(2 * closest_numpy.shape[0])

        # indexes[i] の周辺 (indexes[i-1]とindexes[i]の中点から、indexes[i]とindexes[i+1]の中点まで) に対して、
        # indexes[i]の class c の教師ラベル(ブール値)を割り当てる
        for i in np.arange(len(indexes) - 2) + 1:
            start = max(0, (indexes[i - 1] + indexes[i]) // 2)
            stop = min(closest_numpy.shape[0], (indexes[i] + indexes[i + 1]) // 2)
            closest_numpy[start:stop, c] = label_numpy[indexes[i], c]

    return closest_numpy


def label2vector(labels, num_classes=17, framerate=2, version=2):

    vector_size = 90 * 60 * framerate

    label_half1 = np.zeros((vector_size, num_classes))
    label_half2 = np.zeros((vector_size, num_classes))

    _, _, event_dict, _ = getMetaDataTask("caption", "SoccerNet", version)

    for annotation in labels["annotations"]:

        time = annotation["gameTime"]
        event = annotation["label"]

        half = int(time[0])

        minutes = int(time[-5:-3])
        seconds = int(time[-2::])
        frame = framerate * (seconds + 60 * minutes)

        if event not in event_dict:
            continue
        label = event_dict[event]

        value = 1
        if "visibility" in annotation.keys():
            if annotation["visibility"] == "not shown":
                value = -1

        if half == 1:
            frame = min(frame, vector_size - 1)
            label_half1[frame][label] = value

        if half == 2:
            frame = min(frame, vector_size - 1)
            label_half2[frame][label] = value

    return label_half1, label_half2


def predictions2vector(predictions, num_classes=17, version=2, framerate=2):

    vector_size = 90 * 60 * framerate

    prediction_half1 = np.zeros((vector_size, num_classes)) - 1
    prediction_half2 = np.zeros((vector_size, num_classes)) - 1

    _, _, event_dict, _ = getMetaDataTask("caption", "SoccerNet", version)

    for annotation in predictions["predictions"]:

        time = int(annotation["position"])
        event = annotation["label"]

        half = int(annotation["half"])

        frame = int(framerate * (time / 1000))

        if event not in event_dict:
            continue
        label = event_dict[event]

        value = annotation["confidence"]

        if half == 1:
            frame = min(frame, vector_size - 1)
            prediction_half1[frame][label] = value

        if half == 2:
            frame = min(frame, vector_size - 1)
            prediction_half2[frame][label] = value

    return prediction_half1, prediction_half2


def valid_probability(value):
    fvalue = float(value)
    if fvalue <= 0 or fvalue > 1:
        raise argparse.ArgumentTypeError(
            f"{value} is not a valid probability between 0 and 1"
        )
    return fvalue


def generate_naive_spotting_prediction(
    algo_name,
    fixed_interval=59,
    num_classes=1,
    split="test",
    save_predictions=False,
):
    list_games = getListGames(split=split, task="caption")

    detections_numpy = []
    for game in tqdm(list_games):
        prediction_half1 = generate_naive_spotting_vector(
            average_comment_interval=fixed_interval
        )
        prediction_half2 = generate_naive_spotting_vector(
            average_comment_interval=fixed_interval
        )

        if save_predictions:
            save_path = os.path.join(
                os.environ.get("MODEL_DIR"),
                "models",
                algo_name,
                f"outputs/{split}",
                game,
                "results_spotting.json",
            )
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            generate_naive_spotting_prediction_json(
                game, [prediction_half1, prediction_half2], save_path
            )

        detections_numpy.append(prediction_half1)
        detections_numpy.append(prediction_half2)

    return detections_numpy


def generate_naive_commentary_prediction(
    game_list,
    algo_name,
    fixed_interval_1=59,
    fixed_interval_2=80,
    split="test",
    save_predictions=False,
):

    detections_numpy = []
    for game in tqdm(game_list):
        prediction_half1 = generate_naive_spotting_tensor(
            average_comment_interval_1=fixed_interval_1,
            average_comment_interval_2=fixed_interval_2,
        )
        prediction_half2 = generate_naive_spotting_tensor(
            average_comment_interval_1=fixed_interval_1,
            average_comment_interval_2=fixed_interval_2,
        )
        if save_predictions:
            save_path = os.path.join(
                os.environ.get("MODEL_DIR"),
                "models",
                algo_name,
                f"outputs/{split}",
                game,
                "results_spotting.json",
            )
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            generate_naive_spotting_prediction_json(
                game, [prediction_half1, prediction_half2], save_path
            )

        detections_numpy.append(prediction_half1)
        detections_numpy.append(prediction_half2)

    return detections_numpy


def generate_naive_spotting_tensor(
    average_comment_interval_1=59,
    start_position_1=0,
    average_comment_interval_2=59,
    start_position_2=0,
):
    seconds = 45 * 60
    spot_probability = np.zeros((seconds, 2))
    # probability は 1 か 0 のみ
    spot_probability[start_position_1::average_comment_interval_1, 0] = 1
    spot_probability[start_position_2::average_comment_interval_2, 1] = 1
    return spot_probability


def generate_naive_spotting_vector(
    average_comment_interval=59,
    start_position=0,
):
    seconds = 45 * 60
    spot_probability = np.zeros(seconds)
    # probability は 1 か 0 のみ
    spot_probability[start_position::average_comment_interval] = 1
    return spot_probability


def generate_naive_spotting_prediction_json(game, predictions, save_path):
    json_data = {
        "predictions": [],
    }

    def get_spots(prediction):
        # prediction: [..., [confidence_class_1_i, confidence_class_2_i], ...]
        # return [..., [index_i, confidence_i], ...]
        spots = []
        for i, confidences in enumerate(prediction):
            for class_i, confidence_i in enumerate(confidences):
                if confidence_i > 0:
                    spots.append([i, confidence_i, class_i])
        return spots

    for half, prediction_numpy in enumerate(predictions):
        spots = get_spots(prediction_numpy)
        for spot in spots:
            frame_index = int(spot[0])
            confidence = spot[1]
            class_i = spot[2]

            seconds = int((frame_index) % 60)
            minutes = int((frame_index) // 60)

            prediction_data = dict()
            prediction_data["gameTime"] = (
                f"{half+1} - {int(minutes):02d}:{int(seconds):02d}"
            )
            prediction_data["label"] = "comments"
            prediction_data["category"] = class_i
            prediction_data["position"] = str(int((frame_index) * 1000))
            prediction_data["half"] = str(half + 1)
            prediction_data["confidence"] = str(confidence)
            json_data["predictions"].append(prediction_data)

    json_data["predictions"] = sorted(
        json_data["predictions"],
        key=lambda x: (int(x["half"]), int(x["position"])),
    )
    json_data["game"] = game

    with open(save_path, "w") as output_file:
        json.dump(json_data, output_file, indent=4)

    return json_data


if __name__ == "__main__":
    import pprint
    import tap
    from typing import Literal, Optional
    import warnings

    class Args(tap.Tap):
        type: Optional[Literal["spotting", "commentary", "commentary_gold"]] = (
            "spotting"
        )
        fixed_interval: Optional[int] = None
        fixed_interval_2: Optional[int] = None

    args = Args().parse_args()

    if args.type == "spotting":
        # evaluate naive spotting
        assert (
            args.fixed_interval is not None
        ), "fixed_interval is required for spotting"

        split = "test"
        fixed_interval = args.fixed_interval
        algo_name = f"naive_fixed_interval_{fixed_interval}"

        Predictios_path = os.path.join(
            os.getenv("MODEL_DIR"), "models", algo_name, "outputs", split
        )
        SoccerNet_path = "/raid_elmo/home/lr/moriy/SoccerNet"

        _ = generate_naive_spotting_prediction(
            algo_name=algo_name,
            fixed_interval=fixed_interval,
            split=split,
            save_predictions=True,
        )

        for metric in ["loose", "medium", "tight"]:
            print(f"{metric} mAP")
            results = evaluate(
                SoccerNet_path,
                Predictios_path,
                prediction_file="results_spotting.json",
                split=split,
                version=2,
                framerate=1,
                metric=metric,
            )
            pprint.pprint(results)
            print("##################")
    elif args.type == "commentary":
        from dataset import CommentaryClipsTesting

        assert (args.fixed_interval is not None) and (
            args.fixed_interval_2 is not None
        ), "both fixed_interval and fixed_interval_2 are required for commentary"

        split = "test"
        fixed_interval_1 = args.fixed_interval
        fixed_interval_2 = args.fixed_interval_2
        algo_name = f"commentary_naive_fixed_interval_classA{fixed_interval_1}_classB{fixed_interval_2}"
        Predictios_path = os.path.join(
            os.getenv("MODEL_DIR"), "models", algo_name, "outputs", split
        )
        SoccerNet_path = "/raid_elmo/home/lr/moriy/SoccerNet"

        ## load label list
        labels_list = []
        commentary_dataset = CommentaryClipsTesting(
            path=SoccerNet_path,
            split=split,
            framerate=1,
            features="baidu_soccer_embeddings.npy",
        )
        game_list = []
        for i in range(len(commentary_dataset)):
            (game, feat_half1, feat_half2, label_half1, label_half2) = (
                commentary_dataset[i]
            )

            labels_list.append(label_half1)
            labels_list.append(label_half2)
            game_list.append(game)

        # generate prediction
        predictions_list = generate_naive_commentary_prediction(
            game_list,
            algo_name=algo_name,
            fixed_interval_1=fixed_interval_1,
            fixed_interval_2=fixed_interval_2,
            split=split,
            save_predictions=True,
        )

        # check the size of the predictions_list and labels_list
        assert len(predictions_list) == len(
            labels_list
        ), f"{len(predictions_list)=} ≠ {len(labels_list)=}"

        # label_vector は prediction vectorより長くあるべき
        for idx, (label, prediction) in enumerate(zip(labels_list, predictions_list)):
            if len(label) < len(prediction):
                warnings.warn(
                    f"game: {game_list[idx // 2]}\n"
                    f"{len(label)=} < {len(prediction)=}\n"
                    "label vector should be longer than prediction vector\n"
                    "label vector will be recipt auto-padding"
                )
                labels_list[idx] = np.pad(
                    label,
                    (0, len(prediction) - len(label)),
                    "constant",
                    constant_values=0,
                )

        for metric in ["loose", "medium", "tight"]:
            print(f"{metric} mAP")
            results = evaluate_commentary_spotting(
                labels_list,
                predictions_list,
                metric=metric,
            )
            pprint.pprint(results)
            print("##################")

    elif args.type == "commentary_gold":
        from dataset import CommentaryClipsTesting

        """
        testデータのラベルを使から、教師 results_spotting.json を生成する
        """
        func = generate_naive_spotting_prediction_json
        SoccerNet_path = "/raid_elmo/home/lr/moriy/SoccerNet"
        split = "test"
        commentary_dataset = CommentaryClipsTesting(
            path=SoccerNet_path,
            split=split,
            framerate=1,
            features="baidu_soccer_embeddings.npy",
        )
        for i in range(len(commentary_dataset)):
            (game, feat_half1, feat_half2, label_half1, label_half2) = (
                commentary_dataset[i]
            )
            save_path = os.path.join(
                os.environ.get("MODEL_DIR"),
                "models",
                "commentary_gold",
                f"outputs/{split}",
                game,
                "results_spotting.json",
            )
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            func(game, [label_half1, label_half2], save_path)
