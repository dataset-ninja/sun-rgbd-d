import os, glob
import numpy as np
import scipy.io
import shutil
from collections import defaultdict
import supervisely as sly
from supervisely.io.fs import (
    file_exists,
    get_file_name,
    get_file_name_with_ext,
    get_file_size,
)
from tqdm import tqdm

import src.settings as s
from dataset_tools.convert import unpack_if_archive


def convert_and_upload_supervisely_project(
    api: sly.Api, workspace_id: int, project_name: str
) -> sly.ProjectInfo:
    # Possible structure for bbox case. Feel free to modify as you needs.

    dataset_path = "/home/alex/DATASETS/IMAGES/SUN RGB-D"
    bboxes_path = "/home/alex/DATASETS/IMAGES/SUN RGB-D/SUNRGBDMeta2DBB_v2.mat"

    group_tag_name = "im_id"
    scene_file = "scene.txt"
    mask_file = "seg.mat"
    image_folder = "image"
    depth_folder = "depth"
    depth_bfx_folder = "depth_bfx"
    images_ext = ".jpg"
    depth_suffix = "_abs.png"

    batch_size = 1

    ds_name = "train"


    def create_ann(image_path):
        global meta
        labels = []
        tags = []

        im_id_value = image_path.split("/")[-3] + get_file_name(image_path)
        group_tag = sly.Tag(group_tag_meta, value=im_id_value)
        tags.append(group_tag)

        image_np = sly.imaging.image.read(image_path)[:, :, 0]
        img_height = image_np.shape[0]
        img_wight = image_np.shape[1]

        sensor_value = folder_to_sensor[image_path.split("/")[7]]
        sensor_tag = sly.Tag(sensor_meta, value=sensor_value)
        tags.append(sensor_tag)

        sub_ds_value = image_path.split("/")[8]
        sub_ds_tag = sly.Tag(sub_ds_meta, value=sub_ds_value)
        tags.append(sub_ds_tag)

        scene_path = image_path.replace(
            image_folder + "/" + get_file_name_with_ext(image_path), scene_file
        )
        with open(scene_path) as f:
            content = f.read().split("\n")
            scene_value = content[0]
            scene_tag = sly.Tag(scene_meta, value=scene_value)
            tags.append(scene_tag)

        mask_path = image_path.replace(
            image_folder + "/" + get_file_name_with_ext(image_path), mask_file
        )

        mat = scipy.io.loadmat(mask_path)
        full_mask = mat["seglabel"]
        pixels = np.unique(full_mask)
        if mat["names"].shape[0] == 1:
            names = [name[0] for name in mat["names"][0]]
        else:
            names = [name[0][0] for name in mat["names"]]

        pixel_to_class = {}
        for pixel in pixels[1:]:
            class_name = names[pixel - 1]
            obj_class = meta.get_obj_class(class_name)
            if obj_class is None:
                obj_class = sly.ObjClass(class_name, sly.AnyGeometry)
                pixel_to_class[pixel] = obj_class
                meta = meta.add_obj_class(obj_class)
                api.project.update_meta(project.id, meta.to_json())
            else:
                pixel_to_class[pixel] = obj_class

            mask = full_mask == pixel
            bitmap = sly.Bitmap(data=mask)
            label = sly.Label(bitmap, obj_class)
            labels.append(label)

        bboxes_data = image_path_to_bboxes[image_path]

        for curr_bbox_data in bboxes_data:
            class_name, bbox = curr_bbox_data
            obj_class = meta.get_obj_class(class_name)
            left = int(bbox[0])
            right = left + int(bbox[2])
            top = int(bbox[1])
            bottom = top + int(bbox[3])
            rectangle = sly.Rectangle(top=top, left=left, bottom=bottom, right=right)
            label = sly.Label(rectangle, obj_class)
            labels.append(label)

        return sly.Annotation(img_size=(img_height, img_wight), labels=labels, img_tags=tags)


    sensor_meta = sly.TagMeta(
        "sensor",
        sly.TagValueType.ONEOF_STRING,
        possible_values=["kinect v1", "kinect v2", "asus xtion", "intel realsense"],
    )
    folder_to_sensor = {
        "kv1": "kinect v1",
        "kv2": "kinect v2",
        "realsense": "intel realsense",
        "xtion": "asus xtion",
    }

    sub_ds_meta = sly.TagMeta("sub ds", sly.TagValueType.ANY_STRING)
    scene_meta = sly.TagMeta("scene", sly.TagValueType.ANY_STRING)

    group_tag_meta = sly.TagMeta(group_tag_name, sly.TagValueType.ANY_STRING)

    obj_classes = []
    all_names = []

    image_path_to_bboxes = defaultdict(list)
    mat_bbox = scipy.io.loadmat(bboxes_path)["SUNRGBDMeta2DBB"][0]
    for curr_im_bboxes in mat_bbox:
        image_path = curr_im_bboxes[0][0]
        image_name = curr_im_bboxes[-2][0]
        curr_im_path = os.path.join(dataset_path, image_path, image_folder, image_name)
        for bboxes in curr_im_bboxes[1]:
            for bbox in bboxes:
                class_name = bbox[2][0].replace("_", " ")
                if class_name not in all_names:
                    all_names.append(class_name)
                    obj_class = sly.ObjClass(class_name, sly.AnyGeometry)
                    obj_classes.append(obj_class)
                image_path_to_bboxes[curr_im_path].append([class_name, bbox[1][0]])

    project = api.project.create(workspace_id, project_name, change_name_if_conflict=True)


    meta = sly.ProjectMeta(
        obj_classes=obj_classes,
        tag_metas=[sensor_meta, sub_ds_meta, scene_meta, group_tag_meta],
    )
    api.project.update_meta(project.id, meta.to_json())
    api.project.images_grouping(id=project.id, enable=True, tag_name=group_tag_name)


    dataset = api.dataset.create(project.id, ds_name, change_name_if_conflict=True)


    images_pathes = glob.glob(dataset_path + "/*/*/*/*/image/*.jpg") + glob.glob(
        dataset_path + "/*/*/*/*/*/*/image/*.jpg"
    )

    progress = sly.Progress("Create dataset {}".format(ds_name), len(images_pathes))

    for images_pathes_batch in sly.batched(images_pathes, batch_size=batch_size):
        anns_batch = []
        img_names_batch = []
        all_images_pathes_batch = []
        for image_path in images_pathes_batch:
            depth_path = image_path.replace(image_folder, depth_folder).replace(
                images_ext, depth_suffix
            )
            if file_exists(depth_path) is False:
                depth_path = depth_path.replace(depth_suffix, ".png")
            depth_bfx_path = image_path.replace(image_folder, depth_bfx_folder).replace(
                images_ext, depth_suffix
            )
            if file_exists(depth_bfx_path) is False:
                depth_bfx_path = depth_bfx_path.replace(depth_suffix, ".png")
            full_im_name = image_path.split("/")[-3] + get_file_name_with_ext(image_path)
            full_depth_name = image_path.split("/")[-3] + get_file_name_with_ext(depth_path)
            full_depth_bfx_name = image_path.split("/")[-3] + get_file_name(depth_bfx_path) + "_bfx.png"

            img_names_batch.extend([full_im_name, full_depth_name, full_depth_bfx_name])
            all_images_pathes_batch.extend([image_path, depth_path, depth_bfx_path])

            ann = create_ann(image_path)
            anns_batch.extend([ann, ann, ann])

        img_infos = api.image.upload_paths(dataset.id, img_names_batch, all_images_pathes_batch)
        img_ids = [im_info.id for im_info in img_infos]

        api.annotation.upload_anns(img_ids, anns_batch)

        progress.iters_done_report(len(images_pathes_batch))

    # ===============================================================================================


    def create_ann_test(image_path):
        tags = []

        im_id_value = get_file_name(image_path)
        group_tag = sly.Tag(group_tag_meta, value=im_id_value)
        tags.append(group_tag)

        image_np = sly.imaging.image.read(image_path)[:, :, 0]
        img_height = image_np.shape[0]
        img_wight = image_np.shape[1]

        return sly.Annotation(img_size=(img_height, img_wight), img_tags=tags)


    ds_name = "test"
    dataset_path = "/home/alex/DATASETS/IMAGES/SUN RGB-D/SUNRGBDLSUNTest/SUNRGBDv2Test"

    dataset = api.dataset.create(project.id, ds_name, change_name_if_conflict=True)

    images_pathes = glob.glob(dataset_path + "/*/*/image/*.jpg")

    progress = sly.Progress("Create dataset {}".format(ds_name), len(images_pathes))

    for images_pathes_batch in sly.batched(images_pathes, batch_size=batch_size):
        anns_batch = []
        img_names_batch = []
        all_images_pathes_batch = []
        for image_path in images_pathes_batch:
            depth_path = image_path.replace(image_folder, depth_folder).replace(images_ext, ".png")
            full_im_name = get_file_name_with_ext(image_path)
            full_depth_name = get_file_name_with_ext(depth_path)

            img_names_batch.extend([full_im_name, full_depth_name])
            all_images_pathes_batch.extend([image_path, depth_path])

            ann = create_ann_test(image_path)
            anns_batch.extend([ann, ann])

        img_infos = api.image.upload_paths(dataset.id, img_names_batch, all_images_pathes_batch)
        img_ids = [im_info.id for im_info in img_infos]

        api.annotation.upload_anns(img_ids, anns_batch)
    return project
