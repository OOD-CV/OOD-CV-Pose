## OOD-CV Data

Enhancing the robustness of vision algorithms in real-world scenarios is challenging. One reason is that existing robustness benchmarks are limited, as they either rely on synthetic data or ignore the effects of individual nuisance factors. We present OOD-CV, a benchmark dataset that includes out-of-distribution examples of 10 object categories in terms of pose, shape, texture, context and the weather conditions, and enables benchmarking of models for image classification, object detection, and 3D pose estimation.

[[`Paper`](https://arxiv.org/abs/2111.14341)] [[`Webpage`](http://www.ood-cv.org)] [[`BibTeX`](#citing-ood-cv)]

### Setup Environment

```sh
conda create -n oodcv python=3.9
conda activate oodcv
conda install numpy scipy pillow
pip install wget gdown pyyaml BboxTools opencv-python scikit-image
```


## 3D Object Pose Estimation Tutorial

3D object pose estimation aims to predict the three rotation parameters (azimuth, elevation, in-plane rotation) of an object relative to the camera. In this tutorial, we focus on category-level pose estimation, where the model should discriminative features of 3D viewpoint and generalize to unseen objects in the same category (e.g., aeroplane, boat, car).

![PASCAL3D](https://cvgl.stanford.edu/projects/pascal3d+/pascal3d.png)

### Datasets

**PASCAL3D+ [1].** PASCAL3D+ contains 12 man-made object categories
with 3D pose annotations and 3D meshes for each category respectively. It is often split into a training set with 11045 images and validation set with 10812 images.

**ObjectNet3D [2].** ObjectNet3D is another 3D pose estimation benchmark that contains 100 different categories with 3D meshes, it contains a total of 17101 training samples and 19604 testing samples, including 3556 occluded or truncated testing samples.

**OOD-CV [3].** OOD-CV extends the PASCAL3D+ dataset and investigates models' robustness to out-of-distribution data in terms of pose, shape, texture, context weather, and occlusion.

### Previous Methods

**Classification-based baseline.** Tulsiani et al. [4] formulated the 3D object pose estimation as a bin classification problem. Usually a ResNet-50 model is used as the backbone and two versions are considered, i.e., Res50-General and Res50-Specific depending on if the classification is performed for each object cateogry.

**StarMap [5].** StarMap is a keypoint-based method where they proposed a category-agnostic keypoint representation, which combines a multi-peak heatmap (StarMap) for all the keypoints and their corresponding features as 3D locations in the canonical viewpoint. The proposed approach can be used to solve 3D object pose estimation by solving a perspective-n-points (PnP) problem.

**NeMo [6].** Neural mesh models (NeMo) proposed to learn a generative model of neural features at each vertex of a dense 3D mesh. Then the 3D object pose estimation problem can be solved by analysis-by-synthesis where we minimize the reconstruction error between NeMo and the feature representation of the target image.

### A Baseline Approach

In this section, we will walk through the code and train a classification-based baseline model for 3D pose estimation. The code is based on the released code [here](https://github.com/OOD-CV/OOD-CV-Pose).

**Preparing the data.** We first preprocess the raw data with the provided code:

```sh
python prepare_ood_cv.py \
    --config config.yaml \
    --workers 6
```


### Train a Baseline Model: ResNet50

```sh
python train_baseline.py \
    --config resnet.yaml
```


**Dataloader.** The [provided dataloader](https://github.com/OOD-CV/OOD-CV-Pose/blob/master/dataset.py) would load the processed data for training and testing the pose estimation model. For each item we would first load the image and annotations from local files or cached memory:

```py
name_img, cate = self.file_list[item]
if self.enable_cache and name_img in self.cache.keys():
    sample = copy.deepcopy(self.cache[name_img])
else:
    img = Image.open(os.path.join(self.image_path, f"{name_img}.JPEG"))
    if img.mode != "RGB":
        img = img.convert("RGB")
    annotation_file = np.load(
        os.path.join(self.annotation_path, name_img.split(".")[0] + ".npz"),
        allow_pickle=True,
    )
```

Then the return data is as follows:

```py
label = 0 if len(self.category) == 0 else self.category.index(cate)
sample = {
    "this_name": this_name,
    "cad_index": int(annotation_file["cad_index"]),
    "azimuth": float(annotation_file["azimuth"]),
    "elevation": float(annotation_file["elevation"]),
    "theta": float(annotation_file["theta"]),
    "distance": 5.0,
    "bbox": annotation_file["box_obj"],
    "img": img,
    "original_img": np.array(img),
    "label": label,
}
```

where `label` is the object category, `cad_index` indicates the subtype of the object (a `car` object can be a sedan, sports car, minivan, etc.), and `azimuth`, `elevation`, and `theta` are the ground-truth 3D pose annotations.

**ResNet-50 model.** The model source code is provided [here](https://github.com/OOD-CV/OOD-CV-Pose/blob/master/src/models/resnet.py). During initialization, we modify the loaded model with a pose classification head:

```py
def build(self):
    assert self.backbone in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'], \
        f"Unsupported backbone {self.backbone} for ResNetGeneral"

    model = torchvision.models.__dict__[self.backbone](pretrained=True)
    model.avgpool = nn.AvgPool2d(8, stride=1)
    if self.backbone == 'resnet18':
        model.fc = nn.Linear(512 * 1, self.output_dim)
    else:
        model.fc = nn.Linear(512 * 4, self.output_dim)
    self.model = model.to(self.device)
    if self.checkpoint is not None:
        self.model.load_state_dict(self.checkpoint['state'])

    self.loss_func = construct_class_by_name(**self.training.loss).to(self.device)
    self.optim = construct_class_by_name(
        **self.training.optimizer, params=self.model.parameters())
    self.scheduler = construct_class_by_name(
        **self.training.scheduler, optimizer=self.optim)
```

Since we formulate pose esimtation as a classification problem, we need to convert between poses (floats from $0$ to $2\pi$) and classification bins.

```py
def _get_targets(self, sample):
    azimuth = sample['azimuth'].numpy() / np.pi
    elevation = sample['elevation'].numpy() / np.pi
    theta = sample['theta'].numpy() / np.pi
    theta[theta < -1.0] += 2.0
    theta[theta > 1.0] -= 2.0

    targets = np.zeros((len(azimuth), 3), dtype=np.int32)
    targets[azimuth < 0.0, 0] = self.num_bins - 1 - np.floor(-azimuth[azimuth < 0.0] * self.num_bins / 2.0)
    targets[azimuth >= 0.0, 0] = np.floor(azimuth[azimuth >= 0.0] * self.num_bins / 2.0)
    targets[:, 1] = np.ceil(elevation * self.num_bins / 2.0 + self.num_bins / 2.0 - 1)
    targets[:, 2] = np.ceil(theta * self.num_bins / 2.0 + self.num_bins / 2.0 - 1)

    return torch.from_numpy(targets)

def _prob_to_pose(self, prob):
    pose_pred = np.argmax(prob.reshape(-1, 3, self.num_bins), axis=2).astype(np.float32)
    pose_pred[:, 0] = (pose_pred[:, 0] + 0.5) * np.pi / (self.num_bins / 2.0)
    pose_pred[:, 1] = (pose_pred[:, 1] - self.num_bins / 2.0) * np.pi / (self.num_bins / 2.0)
    pose_pred[:, 2] = (pose_pred[:, 2] - self.num_bins / 2.0) * np.pi / (self.num_bins / 2.0)
    return pose_pred
```

Once the annotations are converted, training the model is basically optimizing a classifcation loss:

```py
def _train(self, sample):
    img = sample['img'].to(self.device)
    targets = self._get_targets(sample).long().view(-1).to(self.device)
    output = self.model(img)

    loss = construct_class_by_name(**self.training.loss).to(self.device)(
        output.view(-1, self.num_bins), targets)

    self.optim.zero_grad()
    loss.backward()
    self.optim.step()

    self.loss_trackers['loss'].append(loss.item())

    return {'loss': loss.item()}
```

One trick during evaluation is to flip the image and average the two predictions:

```py
def _evaluate(self, sample):
    img = sample['img'].to(self.device)
    output = self.model(img).detach().cpu().numpy()

    img_flip = torch.flip(img, dims=[3])
    output_flip = self.model(img_flip).detach().cpu().numpy()

    azimuth = output_flip[:, :self.num_bins]
    elevation = output_flip[:, self.num_bins:2*self.num_bins]
    theta = output_flip[:, 2*self.num_bins:3*self.num_bins]
    output_flip = np.concatenate([azimuth[:, ::-1], elevation, theta[:, ::-1]], axis=1).reshape(-1, self.num_bins * 3)

    output = (output + output_flip) / 2.0

    pose_pred = self._prob_to_pose(output)

    pred = {}
    pred['probabilities'] = output
    pred['final'] = [{'azimuth': pose_pred[0, 0], 'elevation': pose_pred[0, 1], 'theta': pose_pred[0, 2]}]

    return pred
```

**Training and testing.** Now the data and model are ready, we can start the training and testing. The script [here](https://github.com/OOD-CV/OOD-CV-Pose/blob/master/train_baseline.py) demonstrates the steps to train and test on OOD-CV.

```py
logging.info("Start training")
for epo in range(cfg.training.total_epochs):
    num_iterations = int(cfg.training.scale_iterations_per_epoch * len(train_dataloader))
    for i, sample in enumerate(train_dataloader):
        if i >= num_iterations:
            break
        loss_dict = model.train(sample)

    if (epo + 1) % cfg.training.log_interval == 0:
        logging.info(
            f"[Epoch {epo+1}/{cfg.training.total_epochs}] {model.get_training_state()}"
        )

    if (epo + 1) % cfg.training.ckpt_interval == 0:
        torch.save(model.get_ckpt(epoch=epo+1, cfg=cfg.asdict()), os.path.join(cfg.args.save_dir, "ckpts", f"model_{epo+1}.pth"))
        results = evaluate(cfg, val_dataloader, model)
        logging.info(f'[Validation {epo+1}] acc@pi/6={results["acc6"]:.2f} acc@pi/18={results["acc18"]:.2f} mederr={results["mederr"]:.2f}')

    model.step_scheduler()
```

```py
pose_errors = []
for i, sample in enumerate(dataloader):
    pred = model.evaluate(sample)
    _err = pose_error(sample, pred['final'][0])
    pose_errors.append(_err)
pose_errors = np.array(pose_errors)

acc6 = np.mean(pose_errors<np.pi/6) * 100
acc18 = np.mean(pose_errors<np.pi/18) * 100
mederr = np.median(pose_errors) / np.pi * 180
return {'acc6': acc6, 'acc18': acc18, 'mederr': mederr}
```




### References

[1] Xiang, Yu, Roozbeh Mottaghi, and Silvio Savarese. "Beyond pascal: A benchmark for 3d object detection in the wild." In IEEE winter conference on applications of computer vision, pp. 75-82. IEEE, 2014.

[2] Xiang, Yu, Wonhui Kim, Wei Chen, Jingwei Ji, Christopher Choy, Hao Su, Roozbeh Mottaghi, Leonidas Guibas, and Silvio Savarese. "Objectnet3d: A large scale database for 3d object recognition." In Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11-14, 2016, Proceedings, Part VIII 14, pp. 160-176. Springer International Publishing, 2016.

[3] Zhao, Bingchen, Shaozuo Yu, Wufei Ma, Mingxin Yu, Shenxiao Mei, Angtian Wang, Ju He, Alan Yuille, and Adam Kortylewski. "OOD-CV: a benchmark for robustness to out-of-distribution shifts of individual nuisances in natural images." In European Conference on Computer Vision, pp. 163-180. Cham: Springer Nature Switzerland, 2022.

[4] Tulsiani, Shubham, and Jitendra Malik. "Viewpoints and keypoints." In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 1510-1519. 2015.

[5] Zhou, Xingyi, Arjun Karpur, Linjie Luo, and Qixing Huang. "Starmap for category-agnostic keypoint and viewpoint estimation." In Proceedings of the European Conference on Computer Vision (ECCV), pp. 318-334. 2018.

[6] Wang, Angtian, Adam Kortylewski, and Alan Yuille. "Nemo: Neural mesh models of contrastive features for robust 3d pose estimation." arXiv preprint arXiv:2101.12378 (2021).




### Citing OOD-CV

If you find our work useful, please consider giving a star and citation:

```
@inproceedings{zhao2022ood,
  title={OOD-CV: A Benchmark for Robustness to Out-of-Distribution Shifts of Individual Nuisances in Natural Images},
  author={Zhao, Bingchen and Yu, Shaozuo and Ma, Wufei and Yu, Mingxin and Mei, Shenxiao and Wang, Angtian and He, Ju and Yuille, Alan and Kortylewski, Adam},
  booktitle={Computer Vision--ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23--27, 2022, Proceedings, Part VIII},
  pages={163--180},
  year={2022},
  organization={Springer}
}
```
