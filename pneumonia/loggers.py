import io

import telegram_send
import matplotlib.pyplot as plt
from tqdm import tqdm

def send_telegram_message(message):
    telegram_send.send(conf='./telegram.conf', messages=[f'`{message}`'], parse_mode='markdown')

def send_telegram_figure(figure):
    buf = io.BytesIO()
    figure.savefig(buf, format='png')
    buf.seek(0)
    telegram_send.send(conf='./telegram.conf', images=[buf])
    plt.close()

def plot_figure(_):
    plt.show()
    plt.close()

def make_loggers(telegram):
    if telegram:
        return send_telegram_message, send_telegram_figure
    else:
        return tqdm.write, plot_figure
