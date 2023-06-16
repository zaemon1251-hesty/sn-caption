from SoccerNet.Downloader import SoccerNetDownloader as SNdl
mySNdl = SNdl(LocalDirectory="/raid_elmo/home/lr/moriy/SoccerNet/")
mySNdl.downloadGames(
    files=[
        "1_baidu_soccer_embeddings.npy",
        "2_baidu_soccer_embeddings.npy",
        "Labels-caption.json"],
    split=["train", "valid", "test", "challenge"],
    task="caption",
    verbose=1
)

PASSWORD = "s0cc3rn3t"

mySNdl.password = PASSWORD
mySNdl.downloadGames(
    files=["1_224p.mkv", "2_224p.mkv"],
    split=["train", "valid", "test", "challenge"],
    verbose=1,
    task="caption"
)
