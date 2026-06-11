import sapien

OBJECTS = {
    "banana": {
        "collision": "assets/banana/collision.obj",
        "visual": "assets/banana/visual.glb",
        "grasp_pose": sapien.Pose(
            [-0.015109, -0.032888, 0.035],
            [0, 0.998198, 0.02722, 0]
        ),
        "id": [1, 0, 0],
    },

    "hammer": {
        "collision": "assets/urdf_models/models/two_color_hammer/collision.obj",
        "visual": "assets/urdf_models/models/two_color_hammer/textured.obj",
        "grasp_pose": sapien.Pose([0.06647898,  0.00482569, 0.03], [ 0,  0.998198, 0.02722, 0]),
        "id": [0, 1, 0],
    },

    "spoon": {
        "collision": "assets/urdf_models/models/spoon/collision.obj",
        "visual": "assets/urdf_models/models/spoon/textured.obj",
        "grasp_pose": sapien.Pose([-0.04408875, -0.0105613, 0.01], [ 0,  0.998198, 0.02722, 0]),
        "id": [0, 0, 1],
    },
    

    "phillips_screwdriver": {
        "collision": "assets/urdf_models/models/phillips_screwdriver/collision.obj",
        "visual": "assets/urdf_models/models/phillips_screwdriver/textured.obj",
        "grasp_pose": sapien.Pose([-0.0786914, -0.00348671, 0.03], [0, 0.999999, 0.00113446, 0]),
        "id": [1, 1, 0],
    },

    "fork": {
        "collision": "assets/urdf_models/models/fork/collision.obj",
        "visual": "assets/urdf_models/models/fork/textured.obj",
        "grasp_pose": sapien.Pose([-0.0785745, -0.0105613, 0.03], [-0, 0.999704, -0.0243449, 0]), 
        "id": [1, 0, 1],
    },

    "soap": {
        "collision": "assets/urdf_models/models/soap/collision.obj",
        "visual": "assets/urdf_models/models/soap/textured.obj",
        "grasp_pose": sapien.Pose([-0.008123, 0.0014361, 0.03], [-0, 0.993865, -0.110602, 0]), 
        "id": [0, 1, 1],
    },

    "blue_marker": {
        "collision": "assets/urdf_models/models/blue_marker/collision.obj",
        "visual": "assets/urdf_models/models/blue_marker/textured.obj",
        "grasp_pose": sapien.Pose([0.024883, -0.0105613, 0.01], [-0, 0.993865, -0.110602, 0]), 
        "id": [1, 1, 1],
    },

    "mini_claw_hammer_1": {
        "collision": "assets/urdf_models/models/mini_claw_hammer_1/collision.obj",
        "visual": "assets/urdf_models/models/mini_claw_hammer_1/textured.obj",
        "grasp_pose": sapien.Pose([-0.0382963, 0.00145468, 0.01], [0, 0.984808, 0.173648, 0]), 
        "id": [0,0,0],
    },

     
    "toothpaste_1": {
        "collision": "assets/urdf_models/models/toothpaste_1/collision.obj",
        "visual": "assets/urdf_models/models/toothpaste_1/textured.obj",
        "grasp_pose": sapien.Pose([-0.0423645, -0.0036641, 0.03], [-0, 0.998135, -0.0610485, 0]), 
        "id": [0,0,0],
    },

}

