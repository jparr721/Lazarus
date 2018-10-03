'''
This moves data to my drive
so that way I can move data from
the compute cluster to my laptop
without a flash drive
#cloudcomputing
'''

import googleapiclient.errors
import sys
import ast
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from pydrive.files import GoogleDriveFileList
from argparse import ArgumentParser
from os import chdir, listdir, stat

folder_id = '1cVIBMCs5nqWUZZG70zKpNmsNVXWH6JPx'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
            description=('Uploads a folder to drive')
            )
    parser.add_argument(
            '-s',
            type=str,
            help='Point it to the folder you want'
            )
    parser.add_argument(
            '-d',
            type=str,
            help='Where you want it to go'
            )
    parser.add_argument(
            'upload',
            action='store_true',
            help='File path'
            )

    return parser.parse_args()


def authenticate():
    '''Prove yourself'''
    auth = GoogleAuth()
    return GoogleDrive(auth)


def publish_update(drive, folder_id, source):
    '''
    Begin the upload
    '''
