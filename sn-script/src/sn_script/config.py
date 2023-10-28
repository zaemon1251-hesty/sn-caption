from pathlib import Path


class Config:
    base_dir = Path(__file__).parent.parent.parent.parent / "data"
    targets = [
        "SoccerNet/england_epl/2014-2015/2015-02-21 - 18-00 Chelsea 1 - 1 Burnley/",
        "SoccerNet/england_epl/2015-2016/2015-08-23 - 15-30 West Brom 2 - 3 Chelsea/",
        "SoccerNet/england_epl/2016-2017/2016-08-14 - 18-00 Arsenal 3 - 4 Liverpool/",
        "SoccerNet/europe_uefa-champions-league/2014-2015/2014-11-04 - 20-00 Zenit Petersburg 1 - 2 Bayer Leverkusen/",
        "SoccerNet/europe_uefa-champions-league/2015-2016/2015-09-15 - 21-45 Galatasaray 0 - 2 Atl. Madrid/",
        "SoccerNet/europe_uefa-champions-league/2016-2017/2016-09-13 - 21-45 Barcelona 7 - 0 Celtic/",
        "SoccerNet/france_ligue-1/2014-2015/2015-04-05 - 22-00 Marseille 2 - 3 Paris SG/",
        "SoccerNet/france_ligue-1/2016-2017/2017-01-21 - 19-00 Nantes 0 - 2 Paris SG/",
        "SoccerNet/germany_bundesliga/2014-2015/2015-02-21 - 17-30 Paderborn 0 - 6 Bayern Munich/",
        "SoccerNet/germany_bundesliga/2015-2016/2015-08-29 - 19-30 Bayern Munich 3 - 0 Bayer Leverkusen/",
        "SoccerNet/germany_bundesliga/2016-2017/2016-09-10 - 19-30 RB Leipzig 1 - 0 Dortmund/",
        "SoccerNet/italy_serie-a/2014-2015/2015-02-15 - 14-30 AC Milan 1 - 1 Empoli/",
        "SoccerNet/italy_serie-a/2016-2017/2016-08-20 - 18-00 Juventus 2 - 1 Fiorentina/",
        "SoccerNet/spain_laliga/2014-2015/2015-02-14 - 20-00 Real Madrid 2 - 0 Dep. La Coruna/",
        "SoccerNet/spain_laliga/2015-2016/2015-08-29 - 21-30 Barcelona 1 - 0 Malaga/",
        "SoccerNet/spain_laliga/2016-2017/2017-05-21 - 21-00 Malaga 0 - 2 Real Madrid/",
        "SoccerNet/spain_laliga/2019-2020/2019-08-17 - 18-00 Celta Vigo 1 - 3 Real Madrid/",
    ]
