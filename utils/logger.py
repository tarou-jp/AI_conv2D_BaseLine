import logging

def setup_logger(name):
    # ロガーの設定
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # コンソール出力用のハンドラ
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger

# 例としてrootロガーを設定
root_logger = setup_logger('root')
