--------------------------------------------------------------------------------
1. GIOI THIEU
--------------------------------------------------------------------------------

He thong phat hien va ngan chan tan cong prompt hacking trong AI, su dung ket 
hop Rule-based Detection, Machine Learning va Deep Learning (Transformers).

Cac tinh nang chinh:
- Phat hien da thuat toan: 6 mo hinh ML + DistilBERT + Rule-based patterns
- Deep Learning: DistilBERT Transformer voi GPU acceleration (CUDA)
- Da duoc test tren 373K+ mau du lieu thuc te
- Hieu suat cao: F1=0.649 (DistilBERT)

--------------------------------------------------------------------------------
2. THONG TIN DATASET
--------------------------------------------------------------------------------

A. PRODUCTION DATASET (HuggingFace)
   - Nguon: ahsanayub/malicious-prompts
   - Kich thuoc: 373,646 mau
   - Phan chia: 90% train, 10% test
   - Ti le: 24% doc hai, 76% binh thuong
   - Muc dich: Validation cuoi cung va benchmark production

B. CHALLENGING DATASET (Development)
   - Nguon: Cac mau tan cong nang cao tu tao
   - Kich thuoc: 199 mau
   - Ti le: 63% doc hai, 37% binh thuong
   - Dac diem: Jailbreaks tinh vi, edge cases, adversarial examples
   - Muc dich: Phat trien mo hinh va lap nhanh

--------------------------------------------------------------------------------
3. CAI DAT
--------------------------------------------------------------------------------

BUOC 1: Clone repository (Neu chua co source)
-----------------------
   git clone https://github.com/Coah2107/prompt-hacking.git
   cd prompt-hacking

BUOC 2: Khoi dong venv
-----------------------

   A. Tren macOS/Linux:
   --------------------
   source venv/bin/activate

   B. Tren Windows (Command Prompt):
   ---------------------------------
   venv\Scripts\activate.bat

   C. Tren Windows (PowerShell):
   -----------------------------
   .\venv\Scripts\Activate.ps1

BUOC 3: Cai dat cac thu vien can thiet (Neu khong khoi dong duoc venv)
--------------------------------------
   # Cai dat thu vien co ban
   pip install pandas numpy scikit-learn matplotlib seaborn

   # Cai dat thu vien HuggingFace
   pip install datasets

   # Cai dat thu vien luu model
   pip install joblib

   # Cai dat thu vien Deep Learning (bat buoc cho DistilBERT)
   pip install torch transformers

   # Hoac cai dat tat ca cung mot luc
   pip install pandas numpy scikit-learn matplotlib seaborn datasets joblib torch transformers

BUOC 4: Kiem tra cai dat
------------------------
   python -c "import detection_system; print('Cai dat thanh cong!')"

   Neu khong co loi, he thong da san sang su dung.

--------------------------------------------------------------------------------
4. HUONG DAN CHAY DU AN (CHI TIET)
--------------------------------------------------------------------------------

Luu y quan trong: Tat ca cac lenh duoc chay tu thu muc goc cua project 
(prompt-hacking/). Dam bao ban da cd vao thu muc project truoc khi chay.

............................................................................
A. CHAY DETECTOR PIPELINE (detector_pipeline.py)
............................................................................
   Mo ta: Pipeline phat hien chinh, tich hop Rule-based + ML models.
          Huan luyen va danh gia cac mo hinh Machine Learning.

   Cach 1: Chay truc tiep
   ----------------------
   cd detection_system
   python detector_pipeline.py

   Cach 2: Chay tu thu muc goc (khuyen nghi)
   -----------------------------------------
   python -m detection_system.detector_pipeline

   Ket qua:
   - Huan luyen 6 mo hinh ML (SVM, Random Forest, Naive Bayes, v.v.)
   - Luu cac mo hinh da huan luyen vao detection_system/saved_models/
   - In ra ket qua danh gia (F1, Accuracy, Precision, Recall)

............................................................................
B. CHAY TEST PREVENTION SYSTEM (test_prevention_system.py)
............................................................................
   Mo ta: Test toan bo prevention pipeline ket hop voi detection system.
          Kiem tra kha nang loc va ngan chan cac tan cong.

   Chay lenh:
   ----------
   python scripts/test_prevention_system.py

   Hoac:
   -----
   python -m scripts.test_prevention_system

   Ket qua:
   - Test he thong loc dau vao (Input Filter)
   - Test phat hien tan cong bang ML va Deep Learning
   - Hien thi ket qua phan tich cho cac mau prompt khac nhau
   - Thong ke ti le chan thanh cong

............................................................................
C. CHAY COMPLETE SYSTEM TEST (complete_system_test.py)
............................................................................
   Mo ta: Test pipeline bao mat 4 giai doan toi uu:
          Stage 1: Fast Pre-filter (Pattern + Prompt Leaking)
          Stage 2: Semantic Analysis (SVM-based)
          Stage 3: AI Processing (DistilBERT)
          Stage 4: Response Validation

   Chay lenh:
   ----------
   python scripts/complete_system_test.py

   Hoac:
   -----
   python -m scripts.complete_system_test

   Ket qua:
   - Test 4 giai doan bao mat tuan tu
   - Hien thi chi tiet qua trinh xu ly tung prompt
   - Thong ke hieu suat tung giai doan
   - Danh gia tong the he thong

............................................................................
D. CHAY WORKFLOW DEMO (workflow_demo.py)
............................................................................
   Mo ta: Demo day du quy trinh lam viec cua he thong bao mat 4 giai doan.
          Hien thi truc quan cach he thong xu ly cac loai prompt khac nhau.

   Chay lenh:
   ----------
   python scripts/workflow_demo.py

   Hoac:
   -----
   python -m scripts.workflow_demo

   Ket qua:
   - Demo truc quan pipeline bao mat
   - Hien thi chi tiet tung buoc xu ly
   - Test voi nhieu loai tan cong khac nhau
   - Thong ke tong hop hieu suat

............................................................................
E. CAC SCRIPT TEST KHAC
............................................................................
   
   # Test tren challenging dataset
   python scripts/comprehensive_test_suite.py

   # Test tren HuggingFace dataset (373K mau)
   python scripts/huggingface_test.py

   # Tong hop thong tin dataset
   python scripts/dataset_summary.py

   # Benchmark tren nhieu dataset
   python scripts/dataset_benchmark.py

   # Danh gia chi tiet cac mo hinh
   python scripts/evaluate_models.py

............................................................................
F. HUAN LUYEN MO HINH DEEP LEARNING (DISTILBERT)
............................................................................
   Mo ta: Huan luyen mo hinh DistilBERT Transformer cho phat hien tan cong.
          Yeu cau GPU (CUDA) de co hieu suat tot.

   Chay lenh:
   ----------
   python detection_system\models\deep_learning\transformer_detector.py

   Luu y:
   - Can GPU voi CUDA de huan luyen nhanh
   - Thoi gian huan luyen: 30-60 phut tuy cau hinh
   - Model duoc luu tai: detection_system/saved_models/deep_learning/

--------------------------------------------------------------------------------
5. HIEU SUAT CAC MO HINH
--------------------------------------------------------------------------------

Benchmark tren HuggingFace Test Dataset (74,730 mau):

   Hang | Mo hinh              | Loai | F1 Score | Accuracy | Precision | Recall
   -----|----------------------|------|----------|----------|-----------|-------
    1   | DistilBERT           | DL   | 0.6764   | 0.8018   | 0.5484    | 0.8825
    2   | SVM (Fast)           | ML   | 0.4522   | 0.5456   | 0.3153    | 0.7990
    3   | Naive Bayes          | ML   | 0.4289   | 0.6311   | 0.3368    | 0.5902
    4   | Random Forest        | ML   | 0.3826   | 0.2574   | 0.2377    | 0.9806
    5   | SVM                  | ML   | 0.3620   | 0.6886   | 0.3487    | 0.3764
    6   | Logistic Regression  | ML   | 0.2340   | 0.7459   | 0.3999    | 0.1653
    7   | Gradient Boosting    | ML   | 0.1329   | 0.7733   | 0.6482    | 0.0741

Mo hinh tot nhat: DistilBERT (Deep Learning)
   - F1-Score: 0.6491 (+43% so voi ML tot nhat)
   - Accuracy: 78.21%
   - Recall: 85.88% (phat hien phan lon cac tan cong)

--------------------------------------------------------------------------------
6. CAC LOAI TAN CONG DUOC HO TRO
--------------------------------------------------------------------------------

- Prompt Injection
- Jailbreaking
- Social Engineering
- Adversarial Prompts
- System Manipulation
- Role-play Attacks
- Instruction Bypassing
- Context Poisoning

--------------------------------------------------------------------------------
7. CAU TRUC THU MUC
--------------------------------------------------------------------------------

prompt-hacking/
   datasets/                    - Du lieu training va evaluation
      huggingface_dataset_train.csv
      huggingface_dataset_test.csv
      challenging_dataset_train.csv
      challenging_dataset_test.csv
   detection_system/            - He thong phat hien chinh
      config.py                 - Cau hinh he thong
      detector_pipeline.py      - Pipeline phat hien chinh
      features/                 - Trich xuat dac trung
      models/                   - Cac thuat toan phat hien
         rule_based/            - Phat hien dua tren mau
         ml_based/              - Mo hinh Machine Learning
         deep_learning/         - Mo hinh Deep Learning
      evaluation/               - Danh gia hieu suat
      saved_models/             - Cac mo hinh da huan luyen
   prevention_system/           - He thong ngan chan
      filters/                  - Cac bo loc
      validators/               - Cac bo kiem tra
   results/                     - Ket qua danh gia
   docs/                        - Tai lieu ky thuat
   scripts/                     - Cac script test va benchmark
      detector_pipeline.py      - Pipeline phat hien
      test_prevention_system.py - Test he thong ngan chan
      complete_system_test.py   - Test toan bo he thong
      workflow_demo.py          - Demo quy trinh lam viec

--------------------------------------------------------------------------------
8. XU LY LOI THUONG GAP
--------------------------------------------------------------------------------

LOI 1: ModuleNotFoundError - Khong tim thay module
------------------------------------------------------
   Nguyen nhan: Chua cd vao thu muc project hoac chua cai thu vien
   Giai phap:
   - Dam bao dang o thu muc goc: cd prompt-hacking
   - Cai lai thu vien: pip install -r requirements.txt (neu co)
   - Hoac cai thu tuc: pip install pandas numpy scikit-learn torch transformers

LOI 2: FileNotFoundError - Khong tim thay file dataset
------------------------------------------------------
   Nguyen nhan: Chua download hoac dat sai vi tri dataset
   Giai phap:
   - Download dataset tu Google Drive (xem Buoc 2 phan Cai dat)
   - Dam bao cac file .csv nam trong thu muc datasets/

LOI 3: CUDA/GPU khong hoat dong
------------------------------------------------------
   Nguyen nhan: Chua cai dat CUDA hoac driver GPU
   Giai phap:
   - Kiem tra: python -c "import torch; print(torch.cuda.is_available())"
   - Neu False, he thong se tu dong su dung CPU (cham hon)
   - De su dung GPU, cai dat CUDA toolkit va PyTorch phien ban CUDA

LOI 4: Out of Memory khi huan luyen
------------------------------------------------------
   Nguyen nhan: Khong du RAM/VRAM
   Giai phap:
   - Giam batch_size trong config
   - Su dung dataset nho hon (challenging_dataset)
   - Dong cac chuong trinh khac de giai phong RAM