# -*- coding: utf-8 -*-
"""scrapping:
Written by: Eran Tagansky
Usage:
  scrapping.py <bundles_directory_path> <output_directory_path> [<filename_to_tag>...] [--NumOfTries=X]
"""

from bs4 import BeautifulSoup
import os, zipfile, glob, sys
from docopt import docopt

class Tag:
    def __init__(self, name, owner, xxx_todo_changeme):
        (segment, start, end) = xxx_todo_changeme
        self.name = name
        self.owner = owner
        self.segment = segment
        self.start = start
        self.end = end
        self.new_start = start
        self.new_end = end


def endOffset(eOffset, segment, soup):
    """
    This function exist because if the tag is until the end of the segment Atlas.ti decided to write eOffset as -1
    :param eOffset: the end offset according to the project file
    :param segment: the segment according to the project file
    :param soup: a BeautifulSoup object of the file the tag belongs to
    :return: the real end offset of the tag
    """
    if eOffset != -1:
        return eOffset
    else:
        return soup.find('span', {'id': segment}).text.__len__()


#####Program disclaimer: I assumed that each quotation refers to one block(span)

def scrap(bundles_directory_path, output_directory_path, FileNames_to_tag=None, NumOfTries=None):
    if not FileNames_to_tag:
        FileNames_to_tag = glob.glob(bundles_directory_path + '/*.atlproj') + glob.glob(bundles_directory_path + '/*.atlproj.zip')
    else:
        FileNames_to_tag = [bundles_directory_path + '/' + str(FileNames_to_tag[i]) for i in
                            range(len(FileNames_to_tag))]
    if not os.path.exists(bundles_directory_path):
        print('bundles_directory_path not exists')
    if not os.path.exists(output_directory_path):
        print('output_directory_path not exists.creating...')
        os.makedirs(output_directory_path)

    for bundle_path in FileNames_to_tag:
        current_bundle = zipfile.ZipFile(bundle_path)
        project = BeautifulSoup(current_bundle.read('project.aprx'), 'xml')

        # Make all needed directories
        contentID_to_path = {content['id']: content['loc'] for content in project.findAll('content')}
        filename_to_fileContentID = {medium['name']: medium['content'] for medium in project.findAll('medium')}
        tagID_to_tagName = {tag['id']: tag['name'] for tag in project.findAll('tag')}  # check why q and not Q
        quotations_to_tags = {
            quo['id']: [tql['source'] for tql in project.findAll('tagQuotLink') if tql['target'] == quo['id']]
            for quo in project.findAll('quotation')}

        """
        translator = Translator()
        if not NumOfTries:
            tries = 20
        else:
            tries = NumOfTries
        while True:
            try:
                userID_to_username = {user['id']: translator.translate(user['name'], src='iw').text
                                      for user in project.find('users').find_all('user')}
                break
            except:
                tries -= 1
                if tries == 0:
                    return 'need internet connection'
        """
        translate_username = lambda username : username if username != 'ראומה (1)' and username != 'ראומה' else 'Reuma_Heb'
        userID_to_username = {user['id']: translate_username(user['name'])
                              for user in project.find('users').find_all('user')}

        # foreach document create the document as a .txt file where each line is segment including the tags
        for document in project.find_all('document'):
            try:
                document_path = 'contents/' + str(contentID_to_path[filename_to_fileContentID[document['name'] + '.txt']]) + '/content'
                # print document_path
            except:
                print("document \"{0}\" in bundle \"{1}\" not found".format(document['name'],bundle_path.split("/")[-1]))
                continue
            document_soup = BeautifulSoup(current_bundle.read(document_path), 'xml')
            tags = [] # list of all the tags of the file as Tag objects
            for q in document.find_all('quotation'):
                for _tag in quotations_to_tags[q['id']]:
                    tags.append(Tag(
                        tagID_to_tagName[_tag], userID_to_username[q['owner']],
                        (int(float(q.location.segmentedTextLoc['sSegment'])),
                         int(float(q.location.segmentedTextLoc['sOffset'])),
                         endOffset(
                             int(float(q.location.segmentedTextLoc['eOffset'])),
                             q.location.segmentedTextLoc['sSegment'],
                             document_soup))))
            tags = sorted(tags, key=lambda tag: (tag.segment, tag.start, tag.end)) # Important that the list is sorted

            # for each tag find its real offsets in the time we will insert it (Note that the "tags" list is sorted)
            for i in range(tags.__len__()):
                t1 = tags[i]
                tag_len = len('< owner=\'\'>') + len(t1.name) + len(t1.owner)
                for j in range(i + 1, tags.__len__()):
                    t2 = tags[j]
                    if t1.segment == t2.segment:
                        if t2.start >= t1.start:
                            t2.new_start += tag_len
                            t2.new_end += tag_len
                        elif t2.end >= t1.start:
                            t2.new_end += tag_len
                        if t2.start >= t1.end:
                            t2.new_start += 1 + tag_len
                            t2.new_end += 1 + tag_len
                        elif t2.end >= t1.end:
                            t2.new_end += 1 + tag_len


            output_file_path = output_directory_path + '/' + document['name'] + "_tagged" + '.txt'
            # create the file after we found the right offsets for the insertion time
            with open(output_file_path, 'w', encoding='utf-8') as fl:
                for sp in document_soup.find_all('span'):
                    data = ''
                    for content in sp.contents:
                        data += content
                    for i in range(tags.__len__()):
                        if str(tags[i].segment) == sp['id']:
                            # Insert the tag into the segment
                            data = data[:tags[i].new_start] + '<' + str(tags[i].name) + ' owner=\'' + str(
                                tags[i].owner) + '\'>' + data[tags[i].new_start:tags[i].new_end] + '</' + str(
                                tags[i].name) + ' owner=\'' + str(tags[i].owner) + '\'>' + data[tags[i].new_end:]

                    fl.write(data + '\n', )

        current_bundle.close()


if __name__ == '__main__':
    if sys.argv.__len__() == 1:
        # print 'Usage: my_program command --option <argument>'
        print(__doc__)
    else:
        # args = docopt(__doc__, argv='C:\\Users\\erant\\Desktop\\STUDIES\\other\\project\\fwd C:\\Users\\erant\\Desktop\\STUDIES\\other\\project\\tagged')
        args = docopt(__doc__)
        scrap(args['<bundles_directory_path>'], args['<output_directory_path>'], args['<filename_to_tag>'],
              args['--NumOfTries'])
