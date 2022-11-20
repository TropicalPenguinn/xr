import open3d as o3d
import numpy as np
import copy
import random
import time
import matplotlib.pyplot as plt

def draw_registration_results(source_lst, target, transformation_lst):

    target_temp = copy.deepcopy(target)
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp_lst=[]

    for source,transformation in zip(source_lst,transformation_lst):
        source_temp = copy.deepcopy(source)
        source_temp.paint_uniform_color([1, 0.706, 0])
        source_temp.transform(transformation)
        source_temp_lst.append(source_temp)

    source_temp_lst.append(target_temp)

    o3d.visualization.draw_geometries(source_temp_lst,
                                      zoom=0.4559,
                                      front=[0.6452, -0.3036, -0.7011],
                                      lookat=[1.9892, 2.0208, 1.8945],
                                      up=[-0.2779, -0.9482, 0.1556])

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      zoom=0.4559,
                                      front=[0.6452, -0.3036, -0.7011],
                                      lookat=[1.9892, 2.0208, 1.8945],
                                      up=[-0.2779, -0.9482, 0.1556])

def preprocess_point_cloud(pcd, voxel_size):

    pcd_down = pcd.voxel_down_sample(voxel_size)
    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh


def preprocess_point_cloud_source(pcd_down, voxel_size):

    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

def prepare_dataset(voxel_size,source):
   
    source = o3d.io.read_point_cloud(source)
    target = o3d.io.read_point_cloud("stand_sub500000.pcd")

    
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)




    return source, target, source_down, target_down, source_fpfh, target_fpfh


def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):

    distance_threshold = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
           o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ])
    return result

def execute_fast_global_registration(source_down, target_down, source_fpfh,
                                     target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.5

    result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold))
    return result

def refine_registration(source, target, source_fpfh, target_fpfh, voxel_size,result_ransac):
    distance_threshold = voxel_size * 0.4
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return result

def refine_registration_point(source, target, source_fpfh, target_fpfh, voxel_size,result_ransac):
    distance_threshold = voxel_size * 0.4
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000)
        )
    return result

def calculate_coverage(set_lst,flag,size):


    if len(set_lst)==3:
        set1=set_lst[0]
        set2=set_lst[1]
        set3=set_lst[2]
        intersection=len(set1&set2)+len(set1&set3)+len(set2&set3)-2*len(set1&set2&set3)
        print("Coverage rate: {:0.2f}".format(len(flag)/size))
        print("IoU: {:0.2f}".format(intersection/(len(set1|set2|set3))))
        return len(flag)/size
    
    elif len(set_lst)==1:
        #print("Coverage rate: {}".format(len(flag)/size))
        return len(flag)/size
    
    elif len(set_lst)==2:
        set1=set_lst[0]
        set2=set_lst[1]
        intersection=len(set1&set2)
        print("Coverage rate: {:0.2f}".format(len(flag)/size))
        print("IoU: {:0.2f}".format(intersection/(len(set1|set2))))
        return len(flag)/size



def ransac(voxel_size,number):
    down_sample_time=[]
    ransac_time=[]

    for i in range(number):
        source="/home/airlab/catkin_workspace/src/realsense2_description/GT/conversion_result_distance_{}.pcd".format(i)

        # Point Cloud Preparation
        
        down_sample_time.append(0)
        pcd = o3d.io.read_point_cloud(source)
        
        
        # Plane Segmentation
        t=time.time()
        plane_model, inliers = pcd.segment_plane(distance_threshold=0.01,
                                        ransac_n=3,
                                        num_iterations=100)
        [a, b, c, d] = plane_model

        inlier_cloud = pcd.select_by_index(inliers)
        inlier_cloud.paint_uniform_color([1.0, 0, 0])
        outlier_cloud = pcd.select_by_index(inliers, invert=True)


        save="/home/airlab/catkin_workspace/src/realsense2_description/OL/distance_outlier_{}.pcd".format(i)
        o3d.io.write_point_cloud(save,outlier_cloud)
        ransac_time.append(time.time()-t)


    return down_sample_time,ransac_time

# D->R
def ransac_version1(voxel_size,down,number):
    down_sample_time=[]
    ransac_time=[]

    for i in range(number):
        source="/home/airlab/catkin_workspace/src/realsense2_description/GT/conversion_result_distance_{}.pcd".format(i)

        # Point Cloud Preparation
        
        t=time.time()
        pcd = o3d.io.read_point_cloud(source)
        
        pcd=np.asarray(pcd.points)
        r=[i for i in range(pcd.shape[0])]
        r=random.sample(r,down)
        ds=pcd[r]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(ds)


        down_sample_time.append(time.time()-t)

        
        


        # Plane Segmentation
        t=time.time()
        plane_model, inliers = pcd.segment_plane(distance_threshold=0.01,
                                        ransac_n=3,
                                        num_iterations=100)
        [a, b, c, d] = plane_model

        inlier_cloud = pcd.select_by_index(inliers)
        inlier_cloud.paint_uniform_color([1.0, 0, 0])
        outlier_cloud = pcd.select_by_index(inliers, invert=True)


        o3d.visualization.draw_geometries([outlier_cloud],
                                        zoom=0.4559,
                                        front=[0.6452, -0.3036, -0.7011],
                                        lookat=[1.9892, 2.0208, 1.8945],
                                        up=[-0.2779, -0.9482, 0.1556])
        save="/home/airlab/catkin_workspace/src/realsense2_description/OL/distance_outlier_{}.pcd".format(i)
        o3d.io.write_point_cloud(save,outlier_cloud)
        ransac_time.append(time.time()-t)


    return down_sample_time,ransac_time

# R->D
def ransac_version2(voxel_size,down,number):
    down_sample_time=[]
    ransac_time=[]

    for i in range(number):
        source="/home/airlab/catkin_workspace/src/realsense2_description/GT/conversion_result_distance_{}.pcd".format(i)

        # Point Cloud Preparation
        
        pcd = o3d.io.read_point_cloud(source)

        # Plane Segmentation
        t=time.time()
        plane_model, inliers = pcd.segment_plane(distance_threshold=0.01,
                                        ransac_n=3,
                                        num_iterations=100)
        [a, b, c, d] = plane_model

        inlier_cloud = pcd.select_by_index(inliers)
        inlier_cloud.paint_uniform_color([1.0, 0, 0])
        outlier_cloud = pcd.select_by_index(inliers, invert=True)

        ransac_time.append(time.time()-t)

        t=time.time()

        outlier_cloud=np.asarray(outlier_cloud.points)
        r=[i for i in range(outlier_cloud.shape[0])]
        r=random.sample(r,down)
        ds=outlier_cloud[r]
        outlier_cloud = o3d.geometry.PointCloud()
        outlier_cloud.points = o3d.utility.Vector3dVector(ds)

        down_sample_time.append(time.time()-t)


        save="/home/airlab/catkin_workspace/src/realsense2_description/OL/distance_outlier_{}.pcd".format(i)
        o3d.io.write_point_cloud(save,outlier_cloud)
        



    return down_sample_time,ransac_time


def ransac_version3(voxel_size,down,number):
    down_sample_time=[]
    ransac_time=[]

    for i in range(number):
        source="/home/airlab/catkin_workspace/src/realsense2_description/GT/conversion_result_distance_{}.pcd".format(i)

        # Point Cloud Preparation
        
        pcd = o3d.io.read_point_cloud(source)

        # Plane Segmentation
        t=time.time()
        plane_model, inliers = pcd.segment_plane(distance_threshold=0.01,
                                        ransac_n=3,
                                        num_iterations=100)
        [a, b, c, d] = plane_model

        inlier_cloud = pcd.select_by_index(inliers)
        inlier_cloud.paint_uniform_color([1.0, 0, 0])
        outlier_cloud = pcd.select_by_index(inliers, invert=True)

        ransac_time.append(time.time()-t)
        
        t=time.time()

        outlier_cloud=outlier_cloud.uniform_down_sample(down)

        down_sample_time.append(time.time()-t)


        save="/home/airlab/catkin_workspace/src/realsense2_description/OL/distance_outlier_{}.pcd".format(i)
        o3d.io.write_point_cloud(save,outlier_cloud)
        



    return down_sample_time,ransac_time

#random_down_sample(self, sampling_ratio)
def ransac_version4(voxel_size,down,number):
    down_sample_time=[]
    ransac_time=[]

    for i in range(number):
        source="/home/airlab/catkin_workspace/src/realsense2_description/GT/conversion_result_distance_{}.pcd".format(i)

        # Point Cloud Preparation
        
        pcd = o3d.io.read_point_cloud(source)

        # Plane Segmentation
        t=time.time()
        plane_model, inliers = pcd.segment_plane(distance_threshold=0.01,
                                        ransac_n=3,
                                        num_iterations=100)
        [a, b, c, d] = plane_model

        inlier_cloud = pcd.select_by_index(inliers)
        inlier_cloud.paint_uniform_color([1.0, 0, 0])
        outlier_cloud = pcd.select_by_index(inliers, invert=True)
        ransac_time.append(time.time()-t)

        t=time.time()

        outlier_cloud=outlier_cloud.random_down_sample(down)

        down_sample_time.append(time.time()-t)

        
        save="/home/airlab/catkin_workspace/src/realsense2_description/OL/distance_outlier_{}.pcd".format(i)
        o3d.io.write_point_cloud(save,outlier_cloud)
        



    return down_sample_time,ransac_time


#random_down_sample(self, sampling_ratio) D->R
def ransac_version5(voxel_size,down,number):
    down_sample_time=[]
    ransac_time=[]

    for i in range(number):
        source="/home/airlab/catkin_workspace/src/realsense2_description/GT/conversion_result_distance_{}.pcd".format(i)

        # Point Cloud Preparation
        t=time.time()
        pcd = o3d.io.read_point_cloud(source)
        pcd=pcd.random_down_sample(down)
        down_sample_time.append(time.time()-t)

        t=time.time()

        # Plane Segmentation
        plane_model, inliers = pcd.segment_plane(distance_threshold=0.01,
                                        ransac_n=3,
                                        num_iterations=100)
        [a, b, c, d] = plane_model

        inlier_cloud = pcd.select_by_index(inliers)
        inlier_cloud.paint_uniform_color([1.0, 0, 0])
        outlier_cloud = pcd.select_by_index(inliers, invert=True)
        ransac_time.append(time.time()-t)

        
        save="/home/airlab/catkin_workspace/src/realsense2_description/OL/distance_outlier_{}.pcd".format(i)
        o3d.io.write_point_cloud(save,outlier_cloud)
        



    return down_sample_time,ransac_time


def ransac_version6(voxel_size,down,number):
    down_sample_time=[]
    ransac_time=[]

    for i in range(number):
        source="/home/airlab/catkin_workspace/src/realsense2_description/GT/conversion_result_distance_{}.pcd".format(i)

        # Point Cloud Preparation
        t=time.time()
        pcd = o3d.io.read_point_cloud(source)
        pcd=pcd.uniform_down_sample(down)
        down_sample_time.append(time.time()-t)

        t=time.time()

        # Plane Segmentation
        plane_model, inliers = pcd.segment_plane(distance_threshold=0.01,
                                        ransac_n=3,
                                        num_iterations=100)
        [a, b, c, d] = plane_model

        inlier_cloud = pcd.select_by_index(inliers)
        inlier_cloud.paint_uniform_color([1.0, 0, 0])
        outlier_cloud = pcd.select_by_index(inliers, invert=True)
        ransac_time.append(time.time()-t)

        
        save="/home/airlab/catkin_workspace/src/realsense2_description/OL/distance_outlier_{}.pcd".format(i)
        o3d.io.write_point_cloud(save,outlier_cloud)
        



    return down_sample_time,ransac_time


# voxel down
def ransac_version7(voxel_size,number):


    for i in range(number):
        source="/home/airlab/catkin_workspace/src/realsense2_description/GT/conversion_result_distance_{}.pcd".format(i)

        # Point Cloud Preparation
        pcd = o3d.io.read_point_cloud(source)
        pcd=pcd.voxel_down_sample(voxel_size)


        # Plane Segmentation
        plane_model, inliers = pcd.segment_plane(distance_threshold=0.01,
                                        ransac_n=3,
                                        num_iterations=100)
        [a, b, c, d] = plane_model

        inlier_cloud = pcd.select_by_index(inliers)
        inlier_cloud.paint_uniform_color([1.0, 0, 0])
        outlier_cloud = pcd.select_by_index(inliers, invert=True)

        
        save="/home/airlab/catkin_workspace/src/realsense2_description/OL/distance_outlier_{}.pcd".format(i)
        o3d.io.write_point_cloud(save,outlier_cloud)

def register(voxel_size,lst):
    voxel_size=voxel_size
    source_lst=[]
    transform=[]
    flag=set()
    set_lst=[]

    register_time=None
    icp_time=None

    for i in lst:
        s=set()
        source="/home/airlab/catkin_workspace/src/realsense2_description/OL/distance_outlier_{}.pcd".format(i)
        
        # Point Cloud Preparation
        t=time.time()
        source, target, source_down, target_down, source_fpfh, target_fpfh = \
                prepare_dataset(voxel_size,source)
        fpfh_time=time.time()-t
        b=np.asarray(source.points)
        ground_truth=np.asarray(target.points)
        distance=9-0.2*i
        


        t=time.time()
        result_ransac = execute_fast_global_registration(source_down, target_down,
                                            source_fpfh, target_fpfh,voxel_size)

        register_time=time.time()-t


        t=time.time()
        

        #draw_registration_result(source_down,target_down,result_ransac.transformation)
        #draw_registration_result(source,target,result_ransac.transformation)


        result_icp = refine_registration_point(source, target, source_fpfh, target_fpfh,
                            voxel_size,result_ransac)



        #draw_registration_result(source_down,target_down,result_icp.transformation)
        #draw_registration_result(source,target,result_icp.transformation)

        icp_time=time.time()-t
        correspondence_set=np.asarray(result_icp.correspondence_set)
        for correspondence in correspondence_set:
            flag.add(correspondence[-1])
            s.add(correspondence[-1])
        
        set_lst.append(s)
        source_lst.append(source)
        transform.append(result_icp.transformation)
    
    
    
    return distance,calculate_coverage(set_lst,flag,len(target.points)),register_time,icp_time,fpfh_time

def mean(lst):
    sum=0
    for i in lst:
        sum+=i
    return sum/len(lst)

def annot_max(x,y, ax=None):
    x=np.array(x)
    y=np.array(y)
    xmax = x[np.argmax(y)]
    ymax = y.max()
    text= "x={:.3f}, y={:.3f}".format(xmax, ymax)
    if not ax:
        ax=plt.gca()
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=60")
    kw = dict(xycoords='data',textcoords="axes fraction",
              arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
    ax.annotate(text, xy=(xmax, ymax), xytext=(0.94,0.96), **kw)
    return xmax

# down points
def main1():
    # Down Sampling , Ransac
    
    voxel_size=0.13
    down_lst=[1000,1500,2000,2500,3000]
    
    for down in down_lst:
        down_sample_time,ransac_time=ransac_version2(voxel_size,down,43)

        distance=[]
        coverage=[]
        registration_time=0
        icp_time=0
        fpfh_time=0
        # Compute FPFH,Registration,ICP
        for i in range(43):
            lst=[i]
            d,c,r,i,f=register(voxel_size,lst)
            registration_time+=r
            fpfh_time+=f
            icp_time+=i
            coverage.append(c)
            distance.append(d)

        print("DS:{:0.5f},RT:{:0.5f},FT:{:0.5f},RT:{:0.5f},IT:{:0.5f},SUM:{:0.5f}".format(mean(down_sample_time),mean(ransac_time),
        fpfh_time/43,registration_time/43,icp_time/43,
        mean(down_sample_time)+mean(ransac_time)+fpfh_time/43+registration_time/43+icp_time/43))

        plt.plot(distance,coverage)
        annot_max(distance,coverage)

    plt.legend(["1000","1500","2000","2500","3000"])
    plt.savefig('version2.png')
    plt.show()

# original
def main2():
        # Down Sampling , Ransac
    
    voxel_size=0.13

    down_sample_time,ransac_time=ransac(voxel_size,43)

    distance=[]
    coverage=[]
    registration_time=0
    icp_time=0
    fpfh_time=0
    # Compute FPFH,Registration,ICP
    for i in range(43):
        lst=[i]
        d,c,r,i,f=register(voxel_size,lst)
        registration_time+=r
        fpfh_time+=f
        icp_time+=i
        coverage.append(c)
        distance.append(d)

    print("DS:{:0.3f},RT:{:0.3f},FT:{:0.3f},RT:{:0.3f},IT:{:0.3f},SUM:{:0.3f}".format(mean(down_sample_time),mean(ransac_time),
    fpfh_time/43,registration_time/43,icp_time/43,
    mean(down_sample_time)+mean(ransac_time)+fpfh_time/43+registration_time/43+icp_time/43))

    plt.plot(distance,coverage)
    annot_max(distance,coverage)
    plt.savefig('version2.png')
    plt.show()


# uniform_down_sample(self, every_k_points)
def main3():
    # Down Sampling , Ransac
    
    voxel_size=0.13
    down_lst=[100,1000]
    max_point=[]
    for down in down_lst:
        down_sample_time,ransac_time=ransac_version3(voxel_size,down,43)

        distance=[]
        coverage=[]
        registration_time=0
        icp_time=0
        fpfh_time=0
        # Compute FPFH,Registration,ICP
        for i in range(43):
            lst=[i]
            d,c,r,i,f=register(voxel_size,lst)
            registration_time+=r
            fpfh_time+=f
            icp_time+=i
            coverage.append(c)
            distance.append(d)

        print("DS:{:0.5f},RT:{:0.5f},FT:{:0.5f},RT:{:0.5f},IT:{:0.5f},SUM:{:0.5f}".format(mean(down_sample_time),mean(ransac_time),
        fpfh_time/43,registration_time/43,icp_time/43,
        mean(down_sample_time)+mean(ransac_time)+fpfh_time/43+registration_time/43+icp_time/43))

        plt.plot(distance,coverage)
        max=annot_max(distance,coverage)
        max_point.append(max)

    plt.legend(["100","1000"])
    plt.savefig('version2.png')
    plt.show()
    print(max_point)

#random downsample
def main4():
    # Down Sampling , Ransac
    
    voxel_size=0.13
    down_lst=[1/100,1/1000]
    max_point=[]
    for down in down_lst:
        down_sample_time,ransac_time=ransac_version4(voxel_size,down,43)

        distance=[]
        coverage=[]
        registration_time=0
        icp_time=0
        fpfh_time=0
        # Compute FPFH,Registration,ICP
        for i in range(43):
            lst=[i]
            d,c,r,i,f=register(voxel_size,lst)
            registration_time+=r
            fpfh_time+=f
            icp_time+=i
            coverage.append(c)
            distance.append(d)

        print("DS:{:0.5f},RT:{:0.5f},FT:{:0.5f},RT:{:0.5f},IT:{:0.5f},SUM:{:0.5f}".format(mean(down_sample_time),mean(ransac_time),
        fpfh_time/43,registration_time/43,icp_time/43,
        mean(down_sample_time)+mean(ransac_time)+fpfh_time/43+registration_time/43+icp_time/43))

        plt.plot(distance,coverage)
        max=annot_max(distance,coverage)
        max_point.append(max)

    plt.legend(["100","1000"])
    plt.savefig('version2.png')
    plt.show()
    print(max_point)


#random downsample (D->R)
def main5():
    # Down Sampling , Ransac
    
    voxel_size=0.13
    down_lst=[1/1,1/2,1/3,1/4,1/5,1/6,1/7,1/8,1/9,1/10 ]
    max_point=[]
    for down in down_lst:
        down_sample_time,ransac_time=ransac_version5(voxel_size,down,43)

        distance=[]
        coverage=[]
        registration_time=0
        icp_time=0
        fpfh_time=0
        # Compute FPFH,Registration,ICP
        for i in range(43):
            lst=[i]
            d,c,r,i,f=register(voxel_size,lst)
            registration_time+=r
            fpfh_time+=f
            icp_time+=i
            coverage.append(c)
            distance.append(d)

        print("DS:{:0.5f},RT:{:0.5f},FT:{:0.5f},RT:{:0.5f},IT:{:0.5f},SUM:{:0.5f}".format(mean(down_sample_time),mean(ransac_time),
        fpfh_time/43,registration_time/43,icp_time/43,
        mean(down_sample_time)+mean(ransac_time)+fpfh_time/43+registration_time/43+icp_time/43))

        plt.plot(distance,coverage)
        max=annot_max(distance,coverage)
        max_point.append(max)

    plt.legend(["1/1","1/2","1/3","1/4","1/5","1/6","1/7","1/8","1/9","1/10"])
    plt.savefig('version2.png')
    plt.show()
    print(max_point)


# uniform_down_sample(self, every_k_points) (D->R)
def main6():
    # Down Sampling , Ransac
    
    voxel_size=0.13
    down_lst=[1,2,3,4,5,6,7,8,9,10]
    max_point=[]
    for down in down_lst:
        down_sample_time,ransac_time=ransac_version6(voxel_size,down,43)

        distance=[]
        coverage=[]
        registration_time=0
        icp_time=0
        fpfh_time=0
        # Compute FPFH,Registration,ICP
        for i in range(43):
            lst=[i]
            d,c,r,i,f=register(voxel_size,lst)
            registration_time+=r
            fpfh_time+=f
            icp_time+=i
            coverage.append(c)
            distance.append(d)

        print("DS:{:0.5f},RT:{:0.5f},FT:{:0.5f},RT:{:0.5f},IT:{:0.5f},SUM:{:0.5f}".format(mean(down_sample_time),mean(ransac_time),
        fpfh_time/43,registration_time/43,icp_time/43,
        mean(down_sample_time)+mean(ransac_time)+fpfh_time/43+registration_time/43+icp_time/43))

        plt.plot(distance,coverage)
        max=annot_max(distance,coverage)
        max_point.append(max)

    plt.legend(["1","2","3","4","5","6","7","8","9","10"])
    plt.savefig('version2.png')
    plt.show()
    print(max_point)


# just voxel_down
def main7():
        # Down Sampling , Ransac
    
    voxel_size=0.01
    while True:
        t=time.time()
        print(voxel_size)
        if voxel_size>0.5:
            break
        ransac_version7(voxel_size,43)

        distance=[]
        points=[]

        for i in range(43):
            source="/home/airlab/catkin_workspace/src/realsense2_description/OL/distance_outlier_{}.pcd".format(i)
            source = o3d.io.read_point_cloud(source)
            d=9-0.2*i
            distance.append(d)
            points.append(len(source.points))


        plt.plot(distance,points)
        annot_max(distance,points)
        plt.title("{}".format((time.time()-t)/43))
        plt.savefig('version2.png')
        plt.show()

        voxel_size+=0.01
if __name__ == "__main__":

    main7()






    


