import os
import tarfile
import glob
import logging
logging.basicConfig(level=logging.INFO)

def tar_compress(archive_name, source_dir, out_dir):
    """
    input: zip_file name, directory of the file you want to zip
    output: tar.gz file in current directory
    """
    logging.info("compress file {}".format(archive_name))
    with tarfile.open(os.path.join(out_dir, archive_name), "w:gz") as tar:
        os.chdir(source_dir)
        for file_name in glob.glob(source_dir):
            logging.info("   Adding %s " % file_name)
            tar.add(file_name, os.path.basename(file_name))
    tar.close()
    

def errorhandler(filename, path):
    logging.info("Invalid zip type")

def tar_decompress(archive_name, file_path):
    """
    input: zip_file name, destination directory of decompressed files
    output: decompressed files in destination directory
    """
    logging.info("decompress file {}".format(archive_name))
    dest_dir = os.path.join(file_path, archive_name)
    print(dest_dir)
    with tarfile.open(dest_dir, "r") as tar:
        print("extractall")
        tar.extractall(path=file_path)
    tar.close()


class Ziphelper():
    """Purpose is to do switch case for compressing and decompressing"""

    def __init__(self, file_path, output_path, output_filename, ztype='.tar.gz', flags=""):
        self.file_path = file_path
        self.output_path = output_path
        self.output_filename = output_filename
        self.ztype = ztype
        self.flags = flags

    def switch_compress(self):
        logging.info("case is {}".format(self.ztype))
        switcher = {
            ".tar.gz": tar_compress,
        }
        return switcher.get(self.ztype, errorhandler)(self.output_filename + self.ztype, self.file_path, self.output_path)

    def switch_decompress(self):
        logging.info("case is {}".format(self.ztype))
        print("switch decompress")
        switcher = {
            ".tar.gz": tar_decompress,
        }
        return switcher.get(self.ztype, errorhandler)(self.output_filename + self.ztype, self.output_path)
    
    def compressor(self):
        """
        input: file_path of the file you want to zip, name of the output zip file,
                zip type and flags
        output: zip a file in current directory
        """

        logging.info("start compressing")
        logging.info("self.ztype is {}".format(self.ztype))

        # check file_path existance
        if not os.path.exists(self.file_path):
            print("no path")
            logging.error("{} not exist".format(self.file_path))
            return {"error": "no {} file directory".format(self.file_path)}

        # select switch case and do compressing
        self.switch_compress()

    def decompressor(self):
        """
        input: file_path of the file you want to zip, name of the output zip file,
                zip type and flags
        output: zip a file in current directory
        """

        logging.info("start decompressing")
        logging.info("self.ztype is {}".format(self.ztype))

        # check file_path existance
        if not os.path.exists(self.file_path):
            print("path not exist")
            logging.error("{} not exist".format(self.file_path))
            return {"error": "no {} file directory".format(self.file_path)}

        # select switch case and do decompressing
        self.switch_decompress()