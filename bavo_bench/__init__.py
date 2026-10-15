from gymnasium.envs.registration import register


def _register(env_id, entry_point):
    register(
        id=f"bavo-bench/{env_id}",
        entry_point=entry_point,
        # MuJoCo render output may vary slightly across graphics backends.
        nondeterministic=True,
    )


_register(
    "cube-transfer-plate-v1",
    "bavo_bench.env.tasks.cube_transfer_plate.base:CubeTransferPlateEnv",
)
_register(
    "cube-transfer-plate-stage-occlusion-v1",
    "bavo_bench.env.tasks.cube_transfer_plate.stage_occlusion:CubeTransferPlateStageOcclusionEnv",
)
_register(
    "cube-transfer-plate-random-occlusion-v1",
    "bavo_bench.env.tasks.cube_transfer_plate.random_occlusion:CubeTransferPlateRandomOcclusionEnv",
)
_register(
    "block-stack-v1",
    "bavo_bench.env.tasks.block_stack.base:BlockStackEnv",
)
_register(
    "block-stack-stage-occlusion-v1",
    "bavo_bench.env.tasks.block_stack.stage_occlusion:BlockStackStageOcclusionEnv",
)
_register(
    "block-stack-random-occlusion-v1",
    "bavo_bench.env.tasks.block_stack.random_occlusion:BlockStackRandomOcclusionEnv",
)
_register(
    "slide-pickup-v1",
    "bavo_bench.env.tasks.slide_pickup.base:SlidePickupEnv",
)
_register(
    "slide-pickup-stage-occlusion-v1",
    "bavo_bench.env.tasks.slide_pickup.stage_occlusion:SlidePickupStageOcclusionEnv",
)
_register(
    "slide-pickup-random-occlusion-v1",
    "bavo_bench.env.tasks.slide_pickup.random_occlusion:SlidePickupRandomOcclusionEnv",
)
_register(
    "drawer-storage-v1",
    "bavo_bench.env.tasks.drawer_storage.base:DrawerStorageEnv",
)
_register(
    "drawer-storage-stage-occlusion-v1",
    "bavo_bench.env.tasks.drawer_storage.stage_occlusion:DrawerStorageStageOcclusionEnv",
)
_register(
    "drawer-storage-random-occlusion-v1",
    "bavo_bench.env.tasks.drawer_storage.random_occlusion:DrawerStorageRandomOcclusionEnv",
)
_register(
    "dustpan-sweep-v1",
    "bavo_bench.env.tasks.dustpan_sweep.base:DustpanSweepEnv",
)
_register(
    "dustpan-sweep-stage-occlusion-v1",
    "bavo_bench.env.tasks.dustpan_sweep.stage_occlusion:DustpanSweepStageOcclusionEnv",
)
_register(
    "dustpan-sweep-random-occlusion-v1",
    "bavo_bench.env.tasks.dustpan_sweep.random_occlusion:DustpanSweepRandomOcclusionEnv",
)
