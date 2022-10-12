import logging
import sys
import os

log = None


def create_logger(exp_folder, file_name, log_file_only):

    if log_file_only:
        handlers = []
    else:
        handlers = [logging.StreamHandler(sys.stdout)]
    if file_name != '':
        log_path = os.path.join(exp_folder, file_name)
        os.makedirs(os.path.split(log_path)[0], exist_ok=True)
        handlers.append(logging.FileHandler(log_path, mode = 'w'))
    [logging.root.removeHandler(handler) for handler in logging.root.handlers[:]] # remove all existing handlers
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s', handlers=handlers)
    global log
    log = logging.getLogger()

def destroy_logger():
    handlers = log.handlers[:]
    for handler in handlers:
        handler.close()
        log.removeHandler(handler)