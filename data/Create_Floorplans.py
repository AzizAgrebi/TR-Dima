import os
import numpy as np
import cv2
from PIL import Image
import open3d as o3d

R1 = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
theta = np.pi/40 #rad
R2 = np.array([[np.cos(theta), 0., -np.sin(theta)], [0., 1., 0.], [np.sin(theta), 0., np.cos(theta)]])

class PointCloudProcessor:
    def __init__(self, width, height, fx, fy, cx, cy):
        self.cam = o3d.camera.PinholeCameraIntrinsic(width=width, height=height, fx=fx, fy=fy, cx=cx, cy=cy)

    def process_point_clouds(self, paths):
        for path in paths:
            path = "datasets/" + path + "/"
            self._process_depth_images(path)
            
            os.mkdir(path + "images/")
            files = os.listdir(path + "camera/")
            for filename in files:
                pcd = self._create_point_clouds(path, filename)
                pcd = self._modify_point_clouds(pcd)
                pcd = self._project_point_clouds(pcd)
                self._capture_images(path, filename, pcd)
                self._resize_images(path, filename)
                self._improve_images(path, filename)

    def _process_depth_images(self, path):
        files = os.listdir(path + "depth/")
        os.mkdir(path + "depth_images/")
        for filename in files:
            depth_image = Image.fromarray(np.load(path + "depth/" + filename))
            depth_image.save(path + "depth_images/" + filename[0:-4] + ".png")

    def _create_point_clouds(self, path, filename):
        depth_raw = o3d.io.read_image(path + "depth_images/" + "depth_" + filename[7:-5] + ".png")
        color_raw = o3d.io.read_image(path + "camera/" + filename)
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, self.cam)
        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        return pcd

    def _modify_point_clouds(self, pcd):
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)

        mask = points[:, 1] < 0.8
        points2 = points[mask]
        colors2 = colors[mask]
        
        pcd.points = o3d.utility.Vector3dVector(points2)
        pcd.colors = o3d.utility.Vector3dVector(colors2)
        
        coord = np.asarray(pcd.points)
        color = np.zeros(np.asarray(pcd.points).shape)
        
        mask = coord[:, 1] < -0.3
        color[mask] = [0.5, 0.5, 0.5]
        
        mask = coord[:, 1] > 1.0
        color[mask] = [1., 1., 1.]
        
        pcd.colors = o3d.utility.Vector3dVector(color)
        return pcd

    def _project_point_clouds(self, pcd):
        points = np.asarray(pcd.points)
        points[:, 1] = np.where(points[:, 1] >= -0.3, -0.3, points[:, 1])
        pcd.points = o3d.utility.Vector3dVector(points)
        return pcd
    
    def _capture_images(self, path, filename, pcd):
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False)
        pcd.rotate(R1, center=(0, 0, 0))
        pcd.rotate(R2, center=(0, 0, 0))
        vis.add_geometry(pcd)
        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(path + "images/" + "image_" + filename[7:-5] + ".png", True)
        vis.clear_geometries()
        vis.destroy_window()

    def _resize_images(self, path, filename):
        img = cv2.imread(path + "images/" + "image_" + filename[7:-5] + ".png")[40:1000, 480:1480]
        res = Image.fromarray(cv2.resize(img, dsize=(256, 256), interpolation=cv2.INTER_NEAREST))
        res.save(path + "images/" + "image_" + filename[7:-5] + ".png")

    def _improve_images(self, path, filename):
        img = cv2.imread(path + "images/" + "image_" + filename[7:-5] + ".png")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        white_pixels = np.where(gray > 200)
        for i, j in zip(white_pixels[0], white_pixels[1]):
            roi = gray[max(0, i - 4):min(i + 5, gray.shape[0]), max(0, j - 4):min(j + 5, gray.shape[1])]
            num_gray_pixels = np.sum(np.logical_and(roi > 100, roi < 200))
            if num_gray_pixels > 20:
                img[i, j] = [128, 128, 128]
        res = Image.fromarray(img)
        res.save(path + "images/" + "image_" + filename[7:-5] + ".png")