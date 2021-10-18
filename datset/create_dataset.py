import preprocess
import sys
import os
sys.path.append(os.path.join('../', 'src'))
import config
import preprocess

"""
前処理の実行はすべてこのファイルで行う
"""

if __name__ == "__main__":
    preprocess.convert_csv(config.json_train, config.caption_train)
    preprocess.convert_csv(config.json_valid, config.caption_valid, config.caption_test)
    preprocess.image_preprocess(config.caption_train, config.data_train, config.train_image_path, config.train_image)
    preprocess.image_preprocess(config.caption_valid, config.data_valid, config.valid_image_path, config.val_image)
    preprocess.image_preprocess(config.caption_test, config.data_test, config.test_image_path, config.test_image)
