import SimpleITK
import time
import os

import subprocess
import shutil


#from nnunet.inference.predict import predict_from_folder
from predict import predict_from_folder
from nnunet.paths import default_plans_identifier, network_training_output_dir, default_cascade_trainer, default_trainer
from batchgenerators.utilities.file_and_folder_operations import join, isdir
from nnunet.utilities.task_name_id_conversion import convert_id_to_task_name
import torch
from nnunet.inference.ensemble_predictions import merge

import time


class Autopet_baseline():  # SegmentationAlgorithm is not inherited in this class anymore

    def __init__(self):
        """
        Write your own input validators here
        Initialize your model etc.
        """
        # set some paths and parameters
        self.input_path = '/input/'  # according to the specified grand-challenge interfaces
        self.output_path = '/output/images/automated-petct-lesion-segmentation/'  # according to the specified grand-challenge interfaces
        self.nii_path = '/opt/algorithm/nnUNet_raw_data_base/nnUNet_raw_data/Task710_autoPET/imagesTs'
        self.result_path = '/opt/algorithm/nnUNet_raw_data_base/nnUNet_raw_data/Task710_autoPET/result'
        # self.nii_seg_file = 'autoPET_001.nii.gz'
        self.nii_seg_file_list = []
        # pass

    def convert_mha_to_nii(self, mha_input_path, nii_out_path):  #nnUNet specific
        img = SimpleITK.ReadImage(mha_input_path)
        SimpleITK.WriteImage(img, nii_out_path, True)

    def convert_nii_to_mha(self, nii_input_path, mha_out_path):  #nnUNet specific
        img = SimpleITK.ReadImage(nii_input_path)
        SimpleITK.WriteImage(img, mha_out_path, True)

    def check_gpu(self):
        """
        Check if GPU is available
        """
        print('Checking GPU availability')
        is_available = torch.cuda.is_available()
        print('Available: ' + str(is_available))
        print(f'Device count: {torch.cuda.device_count()}')
        if is_available:
            print(f'Current device: {torch.cuda.current_device()}')
            print('Device name: ' + torch.cuda.get_device_name(0))
            print('Device memory: ' + str(torch.cuda.get_device_properties(0).total_memory))

    def load_inputs(self):
        """
        Read from /input/
        Check https://grand-challenge.org/algorithms/interfaces/
        """
        ct_mha_list = os.listdir(os.path.join(self.input_path, 'images/ct/'))
        pet_mha_list = os.listdir(os.path.join(self.input_path, 'images/pet/'))
        uuid_list = [os.path.splitext(ct_mha)[0] for ct_mha in ct_mha_list]
        print ('uuid_list: ', uuid_list)

        for i, (ct_mha, pet_mha) in enumerate(zip(ct_mha_list, pet_mha_list)):
            self.convert_mha_to_nii(os.path.join(self.input_path, 'images/pet/', pet_mha),
                                os.path.join(self.nii_path, 'autoPET_%03d_0000.nii.gz' % i))
            self.convert_mha_to_nii(os.path.join(self.input_path, 'images/ct/', ct_mha),
                                os.path.join(self.nii_path, 'autoPET_%03d_0001.nii.gz' % i))
            self.nii_seg_file_list.append('autoPET_%03d.nii.gz' % i)
        print ('seg_file_list: ', self.nii_seg_file_list)
        return uuid_list

    def write_outputs(self, uuid_list):
        """
        Write to /output/
        Check https://grand-challenge.org/algorithms/interfaces/
        """
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        assert (len(self.nii_seg_file_list) == len(uuid_list))
        for nii_seg_file, uuid in zip(self.nii_seg_file_list, uuid_list):
            self.convert_nii_to_mha(os.path.join(self.result_path, nii_seg_file), os.path.join(self.output_path, uuid + ".mha"))
            print('Output written to: ' + os.path.join(self.output_path, uuid + ".mha"))
        # self.convert_nii_to_mha(os.path.join(self.result_path, self.nii_seg_file), os.path.join(self.output_path, uuid + ".mha"))
        # print('Output written to: ' + os.path.join(self.output_path, uuid + ".mha"))

    def predict(self):
        """
        Your algorithm goes here
        """
        #cproc = subprocess.run(f'nnUNet_predict -i {self.nii_path} -o {self.result_path} -t 710 -m 3d_fullres', shell=True, check=True)
        #os.system(f'nnUNet_predict -i {self.nii_path} -o {self.result_path} -t 710 -m 3d_fullres')
        print("nnUNet segmentation starting!")
        input_folder = self.nii_path
        output_folder = self.result_path
        part_id = 0  #args.part_id
        num_parts = 1  #args.num_parts
        folds = 'None'  # args.folds
        save_npz = True  #args.save_npz
        lowres_segmentations = 'None'  #args.lowres_segmentations
        num_threads_preprocessing = 4  #args.num_threads_preprocessing
        num_threads_nifti_save = 2  # args.num_threads_nifti_save
        disable_tta = False  #args.disable_tta
        overwrite_existing = True  #args.overwrite_existing
        mode = 'normal'  #args.mode
        model = '3d_fullres'  # args.model
        cascade_trainer_class_name = default_cascade_trainer  # args.cascade_trainer_class_name
        disable_mixed_precision = False  #args.disable_mixed_precision

        model_dict = {'nnUNetTrainerV2_S5_D2_W32_LR_1e4_CropSize_192__nnUNetPlansMinSpacing': (0, 1, 2, 3, 4)}
        ratio_list = [1.0]
        step_size = 0.5  #args.step_size
        disable_postprocessing = True
        all_in_gpu = "True"
        crop_size = [192, 192, 192]
        new_threshold = None
        chk = 'model_final_checkpoint'
        task_name = '710'

        if not task_name.startswith("Task"):
            task_id = int(task_name)
            task_name = convert_id_to_task_name(task_id)

        assert model in ["2d", "3d_lowres", "3d_fullres",
                         "3d_cascade_fullres"], "-m must be 2d, 3d_lowres, 3d_fullres or " \
                                                "3d_cascade_fullres"

        if lowres_segmentations == "None":
            lowres_segmentations = None

        if isinstance(folds, list):
            if folds[0] == 'all' and len(folds) == 1:
                pass
            else:
                folds = [int(i) for i in folds]
        elif folds == "None":
            folds = None
        else:
            raise ValueError("Unexpected value for argument folds")

        assert all_in_gpu in ['None', 'False', 'True']
        if all_in_gpu == "None":
            all_in_gpu = None
        elif all_in_gpu == "True":
            all_in_gpu = True
        elif all_in_gpu == "False":
            all_in_gpu = False

        output_dir_list = []
        for trainer, folds in model_dict.items():
            output_dir = join(output_folder, trainer)
            output_dir_list.append(output_dir)
            model_folder_name = join(network_training_output_dir, model, task_name, trainer)
            st_time = time.time()
            predict_from_folder(model_folder_name, input_folder, output_dir, folds, save_npz, num_threads_preprocessing,
                            num_threads_nifti_save, lowres_segmentations, part_id, num_parts, not disable_tta,
                            overwrite_existing=overwrite_existing, mode=mode, overwrite_all_in_gpu=all_in_gpu,
                            mixed_precision=not disable_mixed_precision,
                            step_size=step_size, checkpoint_name=chk, 
                            ratio=None, disable_postprocessing=disable_postprocessing, crop_size=crop_size, new_threshold=new_threshold)
            en_time = time.time()
            print ('Time cost in predict_from_folder: {}'.format(en_time - st_time))

        st_time = time.time()
        merge(output_dir_list, output_folder, 8, override=True, postprocessing_file=None, store_npz=False)
        en_time = time.time()
        print ('Time cost in merge: {}'.format(en_time - st_time))
        for ms_dir in output_dir_list:
            shutil.rmtree(ms_dir)

        print("nnUNet segmentation done!")

    def process(self):
        """
        Read inputs from /input, process with your algorithm and write to /output
        """
        # process function will be called once for each test sample
        # self.check_gpu()
        st_time = time.time()
        print('Start processing')
        uuid_list = self.load_inputs()
        print('Start prediction')
        self.predict()
        print('Start output writing')
        self.write_outputs(uuid_list)
        en_time = time.time()
        print ('Total time: {}'.format(en_time - st_time))


if __name__ == "__main__":
    print("START")
    Autopet_baseline().process()
