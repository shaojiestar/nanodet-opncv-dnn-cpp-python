import cv2
import numpy as np
import argparse

# opencv中dnn模块搭建的nanodet模型
class my_nanodet():
    # 参数为：输入图片大小，置信度阈值，IOU阈值
    def __init__(self, input_shape=320, prob_threshold=0.4, iou_threshold=0.3):
        # 获取coco中的类别名称
        with open('coco.names', 'rt') as f:
            self.classes = f.read().rstrip('\n').split('\n')
        # 获取类别数
        self.num_classes = len(self.classes)
        # 取8，16，32下采样倍数的特征图输入到PANet中进行特征融合
        self.strides = (8, 16, 32)
        # 设置输入图片的尺寸
        self.input_shape = (input_shape, input_shape)
        # ？？？
        self.reg_max = 7
        # 设置置信度阈值
        self.prob_threshold = prob_threshold
        # 设置IOU阈值
        self.iou_threshold = iou_threshold
        # ？？？
        self.project = np.arange(self.reg_max + 1)
        # COCO数据集的均值和方差，变换成1*1*3的向量形式，方便和图片进行运算
        self.mean = np.array([103.53, 116.28, 123.675], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([57.375, 57.12, 58.395], dtype=np.float32).reshape(1, 1, 3)
        if input_shape==320:
            # 320 * 320的图片，用cv2.dnn.readNet函数来加载模型（包括模型结构和参数）
            self.net = cv2.dnn.readNet('nanodet.onnx')
            # 下边这两行是我自己加的，用来调用gpu的
            # PS由于我本身是用pip安装的opencv，所以加了这两行之后也没有调用gpu进行运算
            # 大家可以自己用cmake安装opencv进行测试，看是不是需要加这两行才调用gpu
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        else:
            # 416 * 416的图片
            self.net = cv2.dnn.readNet('nanodet_m.onnx')
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        # 用来保存每个网格的中心点位置
        self.mlvl_anchors = []
        # 在不同的下采样倍数上生成网格点中心坐标
        for i in range(len(self.strides)):
            anchors = self._make_grid((int(self.input_shape[0] / self.strides[i]), int(self.input_shape[1] / self.strides[i])), self.strides[i])
            # 将三种下采样倍数下的所有网格中心点坐标存在mlvl_anchors数组中，形状为[3, ]
            self.mlvl_anchors.append(anchors)
    # 生成网格点中心坐标的具体函数
    def _make_grid(self, featmap_size, stride):
        # 获取一共有多少网格点 PS：同时也是预测时输入特征图的宽高，所以如此命名
        feat_h, feat_w = featmap_size
        # 将每个网格点左上角的坐标（x, y）分别生成
        shift_x = np.arange(0, feat_w) * stride
        shift_y = np.arange(0, feat_h) * stride
        # 利用x和y数据生成网格点坐标矩阵 PS：两个，每个矩阵的大小为[feat_h, feat_w]
        xv, yv = np.meshgrid(shift_x, shift_y)
        # 将[feat_h, feat_w]的两个矩阵平铺成一维的
        xv = xv.flatten()
        yv = yv.flatten()
        # 获取每个网格中心点的x，y坐标，并组合成矩阵
        cx = xv + 0.5 * (stride-1)
        cy = yv + 0.5 * (stride - 1)
        # 将两个[feat_h* feat_w，1]的矩阵合并成[feat_h* feat_w，2]
        return np.stack((cx, cy), axis=-1)
    # softmax函数
    def softmax(self,x, axis=1):
        x_exp = np.exp(x)
        # 如果是列向量，则axis=0
        x_sum = np.sum(x_exp, axis=axis, keepdims=True)
        s = x_exp / x_sum
        return s
    # 对图片进行归一化操作（减均值除方差）
    def _normalize(self, img):   ### c++: https://blog.csdn.net/wuqingshan2010/article/details/107727909
        img = img.astype(np.float32)
        img = (img - self.mean) / self.std
        return img
    # 改变图像的大小，做到锁定高宽比的同时将图片缩放成网络的input_shape
    def resize_image(self, srcimg, keep_ratio=True):
        top, left, newh, neww = 0, 0, self.input_shape[0], self.input_shape[1]
        if keep_ratio and srcimg.shape[0] != srcimg.shape[1]:
            # 求得高宽比
            hw_scale = srcimg.shape[0] / srcimg.shape[1]
            # 如果高宽比大于1的话，就保持网络规定的输入图像大小中高度不变，将宽度变换至原图片纵横比，
            # 并将空白部分用黑色像素填充
            if hw_scale > 1:
                # 获取原图片将要变换的高和宽
                newh, neww = self.input_shape[0], int(self.input_shape[1] / hw_scale)
                # 对原图片进行形状变换
                img = cv2.resize(srcimg, (neww, newh), interpolation=cv2.INTER_AREA)
                # 保持变换后的图片居中输入网络
                left = int((self.input_shape[1] - neww) * 0.5)
                # 此时输入图片的大小为[input_shape[0],小于input_shape[1]]，网络输入的空白部分用黑色填充
                img = cv2.copyMakeBorder(img, 0, 0, left, self.input_shape[1] - neww - left, cv2.BORDER_CONSTANT,
                                         value=0)  # add border
            # 如果高宽比小于1的话，就保持网络规定的输入图像大小中宽度不变，将高度变换至原图片纵横比
            # 并将空白部分用黑色像素填充
            else:
                newh, neww = int(self.input_shape[0] * hw_scale), self.input_shape[1]
                img = cv2.resize(srcimg, (neww, newh), interpolation=cv2.INTER_AREA)
                top = int((self.input_shape[0] - newh) * 0.5)
                img = cv2.copyMakeBorder(img, top, self.input_shape[0] - newh - top, 0, 0, cv2.BORDER_CONSTANT, value=0)
        # 如果输入图片是正方形，就直接缩放至网络输入大小
        else:
            img = cv2.resize(srcimg, self.input_shape, interpolation=cv2.INTER_AREA)
        # 返回进行形状变换之后的图片和它的新宽高，以及它的左上角坐标
        return img, newh, neww, top, left
    # 检测函数
    def detect(self, srcimg):
        # 对图片进行大小变换
        img, newh, neww, top, left = self.resize_image(srcimg)
        # 对图片进行归一化
        img = self._normalize(img)
        # 返回一个4通道的blob作为网络的输入
        blob = cv2.dnn.blobFromImage(img)
        # Sets the input to the network
        self.net.setInput(blob)

        # Runs the forward pass to get output of the output layers
        outs = self.net.forward(self.net.getUnconnectedOutLayersNames())
        # 对网络的输出进行后处理，得到目标框，置信度和类别ID
        det_bboxes, det_conf, det_classid = self.post_process(outs)
        # 在这张新图片上进行目标框的绘制
        drawimg = srcimg.copy()
        # 在图片进行预处理的时候对图片的宽高进行了变换，所以现在要将目标框的坐标变换回去，
        # 使得它在原图上绘制的时候不出错。
        # 求得变换的比例
        ratioh,ratiow = srcimg.shape[0]/newh,srcimg.shape[1]/neww
        # 对于每一个目标框做变换
        for i in range(det_bboxes.shape[0]):
            xmin, ymin, xmax, ymax = max(int((det_bboxes[i,0] - left) * ratiow), 0), max(int((det_bboxes[i,1] - top) * ratioh), 0), min(
                int((det_bboxes[i,2] - left) * ratiow), srcimg.shape[1]), min(int((det_bboxes[i,3] - top) * ratioh), srcimg.shape[0])
            # 绘制预测框
            self.drawPred(drawimg, det_classid[i], det_conf[i], xmin, ymin, xmax, ymax)
        # 返回绘制好的图片
        return drawimg

    # 对网络的输出结果进行后处理
    # 网络输出一共有6个矩阵，分别为8，16，32倍下采样的输出矩阵：
    '''
    (1600, 80):它对应的是40x40的特征图(拉平后是长度为1600的向量，也就是说一共有1600个像素点)
        里的每个像素点在coco数据集的80个类别里的每个类的置信度。
    (1600, 32):它对应的是40x40的特征图(拉平后是长度为1600的向量，也就是说一共有1600个像素点)
        里的每个像素点的检测框的预测偏移量，
        可以看到这个预测偏移量是一个长度为32的向量，它可以分成4份，每份向量的长度为8
    后面4个以此类推即可。
    (400, 80):
    (400, 32):
    (100, 80):
    (100, 32):
    '''
    def post_process(self, preds):
        # 将六个输出分为两拨，第一波1，3，5是不同尺度下每个像素点的类别置信度，
        # 第二波2，4，6是不同尺度下每个像素点的预测偏移量
        cls_scores, bbox_preds = preds[::2], preds[1::2]
        # 获取筛选后的1000目标框，目标置信度和目标类别ID
        det_bboxes, det_conf, det_classid = self.get_bboxes_single(cls_scores, bbox_preds, 1, rescale=False)
        return det_bboxes.astype(np.int32), det_conf, det_classid
    # 获取筛选后的目标框，目标置信度和目标类别ID
    def get_bboxes_single(self, cls_scores, bbox_preds, scale_factor, rescale=False):
        # 用来存放目标框结果和分数
        mlvl_bboxes = []
        mlvl_scores = []
        # 将三个不同级别的预测结果区分开来，逐一进行处理
        for stride, cls_score, bbox_pred, anchors in zip(self.strides, cls_scores, bbox_preds, self.mlvl_anchors):
            # squeeze函数去掉矩阵里维度为1的维度
            # 下面以8倍下采样的结果进行数字推演
            cls_score = cls_score.squeeze()
            bbox_pred = bbox_pred.squeeze()
            # 将预测的偏移量从（1600，32）变为（6400，8）
            # 其实就等同于把每一行切分成4份组成新的矩阵，
            # 然后做softmax变换，把数值归一化到0至1的区间内。
            bbox_pred = self.softmax(bbox_pred.reshape(-1, self.reg_max + 1), axis=1)
            # bbox_pred = np.sum(bbox_pred * np.expand_dims(self.project, axis=0), axis=1).reshape((-1, 4))
            # 将预测结果的（6400，8）与列向量（8，1）进行内积，得到（6400，1）的结果，
            # 在将它变换成（1600，4）.此时1600代表像素点个数，4代表每个像素点距离预测框上下左右距离的预测偏移量
            bbox_pred = np.dot(bbox_pred, self.project).reshape(-1,4)
            # 将预测偏移量与步长相乘，得到每个像素点到预测框四条边的距离
            bbox_pred *= stride

            # nms_pre = cfg.get('nms_pre', -1)
            # 在非极大值抑制环节之前，在1600+400+100个像素点中根据置信度筛选出前1000个
            nms_pre = 1000
            if nms_pre > 0 and cls_score.shape[0] > nms_pre:
                # 获得每个预测框80个类别中最高类别的置信度分数
                max_scores = cls_score.max(axis=1)
                # 对1600+400+100个像素点的最高类别的置信度分数进行排序，
                # 得到置信度分数最高的前1000个预测框的下标
                topk_inds = max_scores.argsort()[::-1][0:nms_pre]
                # 获取这1000个预测框所属的级别（8，16，32）和它们的目标框，类别分数
                anchors = anchors[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                cls_score = cls_score[topk_inds, :]
            # 从到四条边的距离转化成左上角和右下角坐标
            bboxes = self.distance2bbox(anchors, bbox_pred, max_shape=self.input_shape)
            # 保存1000个预测框左上，右下角点坐标和类别分数（这里还是80个类别的分数）
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(cls_score)
        # 将这些数组拼接成一个大数组
        mlvl_bboxes = np.concatenate(mlvl_bboxes, axis=0)
        if rescale:
            mlvl_bboxes /= scale_factor
        mlvl_scores = np.concatenate(mlvl_scores, axis=0)

        bboxes_wh = mlvl_bboxes.copy()
        # 获得预测框的宽高信息
        bboxes_wh[:, 2:4] = bboxes_wh[:, 2:4] - bboxes_wh[:, 0:2]  ####xywh
        # 获得每个预测框的类别信息（80个类别里面置信度分数最高的那个类别）
        classIds = np.argmax(mlvl_scores, axis=1)
        # 获得每个预测框的置信度信息（80个类别里面最高的那个置信度分数）
        confidences = np.max(mlvl_scores, axis=1)  ####max_class_confidence
        # 对1000个预测框进行置信度抑制和IOU抑制
        # 说白了就是置信度分数太低的不要，多个框重合的只要最好的那个框
        indices = cv2.dnn.NMSBoxes(bboxes_wh.tolist(), confidences.tolist(), self.prob_threshold, self.iou_threshold)
        # 这就是要出最后的结果了，返回筛选后的所有目标框（如果有）
        if len(indices)>0:
            mlvl_bboxes = mlvl_bboxes[indices[:, 0]]
            confidences = confidences[indices[:, 0]]
            classIds = classIds[indices[:, 0]]
            return mlvl_bboxes, confidences, classIds
        # 没有的话就返回三个空数组，防止程序报错
        else:
            print('nothing detect')
            return np.array([]), np.array([]), np.array([])
    # 将每个预测框从到四边的距离转化成左上角，右下角坐标
    # 所谓的points数组其实就是每个级别（8，16，32）的坐标矩阵
    # distance就是到预测框四边的距离
    # 想要求得左上角和右下角的x，y坐标
    def distance2bbox(self, points, distance, max_shape=None):
        # 用每个网格中心点x坐标减去到预测框左边的距离得到左上角点的x坐标
        x1 = points[:, 0] - distance[:, 0]
        # 用每个网格中心点y坐标减去到预测框上边的距离得到左上角点的y坐标
        y1 = points[:, 1] - distance[:, 1]
        # 用每个网格中心点x坐标加上到预测框右边的距离得到右下角点的x坐标
        x2 = points[:, 0] + distance[:, 2]
        # 用每个网格中心点y坐标加上到预测框下边的距离得到右下角点的y坐标
        y2 = points[:, 1] + distance[:, 3]
        # 如果设置了最大和最小值，就用clip函数来使得左上角点和右下角点的坐标不超过边界值
        # 这里被设置成0到输入图像的大小
        if max_shape is not None:
            x1 = np.clip(x1, 0, max_shape[1])
            y1 = np.clip(y1, 0, max_shape[0])
            x2 = np.clip(x2, 0, max_shape[1])
            y2 = np.clip(y2, 0, max_shape[0])
        # 返回一个2*2的数组
        return np.stack([x1, y1, x2, y2], axis=-1)
    # 将预测的结果画到原图中去
    def drawPred(self, frame, classId, conf, left, top, right, bottom):
        # Draw a bounding box.
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), thickness=4)

        label = '%.2f' % conf
        label = '%s:%s' % (self.classes[classId], label)

        # Display the label at the top of the bounding box
        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, labelSize[1])
        # cv.rectangle(frame, (left, top - round(1.5 * labelSize[1])), (left + round(1.5 * labelSize[0]), top + baseLine), (255,255,255), cv.FILLED)
        cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2)
        return frame

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgpath', type=str, default='street.png', help="image path")
    parser.add_argument('--input_shape', default=320, type=int, choices=[320, 416], help='input image shape')
    parser.add_argument('--confThreshold', default=0.35, type=float, help='class confidence')
    parser.add_argument('--nmsThreshold', default=0.6, type=float, help='nms iou thresh')
    args = parser.parse_args()

    srcimg = cv2.imread(args.imgpath)
    net = my_nanodet(input_shape=args.input_shape, prob_threshold=args.confThreshold, iou_threshold=args.nmsThreshold)
    import time
    a = time.time()
    srcimg = net.detect(srcimg)
    b = time.time()
    print('waste time', b-a)

    winName = 'Deep learning object detection in OpenCV'
    cv2.namedWindow(winName, cv2.WINDOW_NORMAL)
    cv2.imshow(winName, srcimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
