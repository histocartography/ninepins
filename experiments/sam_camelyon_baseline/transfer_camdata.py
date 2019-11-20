import os
import re

from ftplib import FTP
import concurrent.futures

import histocartography.io.utils as utils

class CamelyonTransfer():
    def __init__(self):

        access_key = os.environ["AWS_ACCESS_KEY_ID"]
        secret_key = os.environ["AWS_SECRET_ACCESS_KEY"]

        self.bucket_name = 'curated-datasets'
        self.s3_connection = utils.get_s3(aws_access_key_id=access_key,
                                     aws_secret_access_key=secret_key)

        ftp_connection = utils.open_ftp_connection('parrot.genomics.cn')

        folders = ['gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/tumor/',
                   'gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/normal/',
                   'gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/testing/images/']

        self.all_files = []
        for folder in folders:
            files = ftp_connection.nlst(folder)
            self.all_files += files

        ftp_connection.close()


    def init_transfer(self):
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
             executor.map(self.threaded_download, self.all_files)
        pass

    def threaded_download(self, image):
        ftp_connection = utils.open_ftp_connection('parrot.genomics.cn')
        filename = os.path.basename(image)
        regex = r"(?:[\w\.]*\/){5}(.*)"
        matches = re.search(regex, os.path.dirname(image))
        dirname =  matches[1]
        print(dirname)
        s3_file_path = f'breast/WSI/{dirname}/{filename}'
        print(f'Transfering {filename}')
        utils.transfer_file_from_ftp_to_s3(self.s3_connection.meta.client,
                                           ftp_connection,
                                           self.bucket_name,
                                           image,
                                           s3_file_path)
        ftp_connection.close()

def main():

    ct = CamelyonTransfer()
    ct.init_transfer()

if __name__ == "__main__":
    main()
