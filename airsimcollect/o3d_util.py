import open3d

def get_extrinsics(vis):
    ctr = vis.get_view_control()
    camera_params = ctr.convert_to_pinhole_camera_parameters()
    return camera_params.extrinsic

def set_view(vis, extrinsics):
    ctr = vis.get_view_control()
    camera_params = ctr.convert_to_pinhole_camera_parameters()
    # print(camera_params.intrinsic.intrinsic_matrix)
    # print(camera_params.extrinsic)
    camera_params.extrinsic = extrinsics
    ctr.convert_from_pinhole_camera_parameters(camera_params)