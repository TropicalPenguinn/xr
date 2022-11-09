import open3d as o3d
import numpy as np
import copy


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


def prepare_dataset(voxel_size,source):
   
    source = o3d.io.read_point_cloud(source)
    target = o3d.io.read_point_cloud("stand_sub.pcd")


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
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result


def refine_registration(source, target, source_fpfh, target_fpfh, voxel_size,result_ransac):
    distance_threshold = voxel_size * 0.4
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return result

def calculate_coverage(set_lst):
    set1=set_lst[0]
    set2=set_lst[1]
    set3=set_lst[2]
    intersection=len(set1&set2)+len(set1&set3)+len(set2&set3)-2*len(set1&set2&set3)
    print("Coverage rate: {:0.2f}".format(len(flag)/101646))
    print("IoU: {:0.2f}".format(intersection/(len(set1|set2|set3))))


def find_optimal_voxel_size():
    
    max=voxel_size=0.01
    while True:
        if voxel_size>0.5:
            break
        
        result=0
        for i in range(3):
             for i in [1,2,3,4,5,6,7,8,9,10,11,12]:
                s=set()
                source="/home/airlab/catkin_workspace/src/realsense2_description/ourlier{}.pcd".format(i)
                

                source, target, source_down, target_down, source_fpfh, target_fpfh = \
                        prepare_dataset(voxel_size,source)

                b=np.asarray(source.points)
                ground_truth=np.asarray(target.points)
                
                result_ransac = execute_global_registration(source_down, target_down,
                                                source_fpfh, target_fpfh,
                                                voxel_size)
                result_icp = refine_registration(source, target, source_fpfh, target_fpfh,
                                    voxel_size,result_ransac)
                result+=result_icp.fitness
        
        result/=36
        if result>max:
            max=result
            print(voxel_size,max)
        voxel_size+=0.01


def register(voxel_size,v1,v2,v3):
    voxel_size=voxel_size
    source_lst=[]
    transform=[]
    flag=set()
    set_lst=[]
    for i in [v1,v2,v3]:
        s=set()
        source="/home/airlab/catkin_workspace/src/realsense2_description/ourlier{}.pcd".format(i)
        
        # Point Cloud Preparation

        source, target, source_down, target_down, source_fpfh, target_fpfh = \
                prepare_dataset(voxel_size,source)

        b=np.asarray(source.points)
        print(b.shape)
        ground_truth=np.asarray(target.points)
        # Plane Segmentation
        #plane_model, inliers = pcd.segment_plane(distance_threshold=0.01,
                                        #ransac_n=3,
                                        #num_iterations=1000)
        #[a, b, c, d] = plane_model

        #inlier_cloud = source.select_by_index(inliers)
        #inlier_cloud.paint_uniform_color([1.0, 0, 0])
        #outlier_cloud = source.select_by_index(inliers, invert=True)
        #o3d.visualization.draw_geometries([outlier_cloud],
                                        #zoom=0.8,
                                        #front=[-0.4999, -0.1659, -0.8499],
                                        #lookat=[2.1813, 2.0619, 2.0999],
                                        #up=[0.1204, -0.9852, 0.1215])
        
        result_ransac = execute_global_registration(source_down, target_down,
                                        source_fpfh, target_fpfh,
                                        voxel_size)
        result_icp = refine_registration(source, target, source_fpfh, target_fpfh,
                            voxel_size)
        correspondence_set=np.asarray(result_icp.correspondence_set)
        for correspondence in correspondence_set:
            flag.add(correspondence[-1])
            s.add(correspondence[-1])
        
        set_lst.append(s)
        source_lst.append(source)
        transform.append(result_icp.transformation)
    
    
    
    calculate_coverage(set_lst)
    draw_registration_results(source_lst,target,transform)


if __name__ == "__main__":

    voxel_size=find_optimal_voxel_size()
    print(voxel_size)

