import torch
# Learning setting
config = dict(setting="supervisedlearning",

              dataset=dict(name="iiitk",
                           datadir="/home/venkat/Projects/Zoho/cords/data/IIIT5K/IIIT5K-Word_V3.0",
                           feature="dss",
                           type="pre-defined"),

              dataloader=dict(shuffle=True,
                              batch_size=20,
                              pin_memory=True),

              model=dict(architecture='CRNN',
                         type='pre-defined',
                        ),
              
              ckpt=dict(is_load=False,
                        is_save=True,
                        dir='results/',
                        save_every=20),
              
              loss=dict(type='CTCLoss',
                        use_sigmoid=False),

              optimizer=dict(type="adam",
                             
                             lr=0.001,
                             ),

              scheduler=dict(type="cosine_annealing",
                             T_max=300),

              dss_strategy=dict(type="full",
                                fraction=0.1,
                                select_every=20),

              train_args=dict(num_epochs=300,
                              device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                              print_every=10,
                              results_dir='results/',
                              print_args=["val_loss", "tst_loss", "time"],
                              return_args=[]
                              )
              )
