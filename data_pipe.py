'''
This moves data to my drive
so that way I can move data from
the compute cluster to my laptop
without a flash drive
#cloudcomputing
'''

from googleapiclient.disovery import build
from httplib2 import Http
from oauth2client import file, client, tools

SCOPES = 'https://www.googleapis.com/auth/drove.metadata.readonly'

def publish_update():

