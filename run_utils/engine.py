
import logging
from enum import Enum
import torch
import tqdm
from itertools import cycle

####
class Events(Enum):
    EPOCH_STARTED = "epoch_started"
    EPOCH_COMPLETED = "epoch_completed"
    STEP_STARTED = "step_started"
    STEP_COMPLETED = "step_completed"
    STARTED = "started"
    COMPLETED = "completed"
    EXCEPTION_RAISED = "exception_raised"


####
class State(object):
    """
    An object that is used to pass internal and 
    user-defined state between event handlers
    """

    def __init__(self):
        # settings propagated from config
        self.logging = None
        self.log_dir = None
        self.log_info = None

        # internal variable
        self.curr_epoch_step = 0  # current step in epoch
        self.curr_global_step = 0  # current global step
        self.curr_epoch = 0  # current global epoch

        # TODO: [LOW] better document this
        # for outputing value that will be tracked per step
        # "scalar" will always be printed out and added to the tensorboard
        # "images" will need dedicated function to process and added to the tensorboard

        # ! naming should match with types supported for serialize
        # TODO: Need way to dynamically adding new types
        self.tracked_step_output = {
            'scalar': {},  # type : {variable_name : variablee_value}
            'image': {},
        }
        # TODO: find way to known which method bind/interact with which value

        self.epoch_accumulated_output = {}  # all output of the current epoch

        # TODO: soft reset for pertain variable for N epochs
        self.run_accumulated_output = []  # of run until reseted

        # holder for output returned after current runstep
        # * depend on the type of training i.e GAN, the updated accumulated may be different
        self.step_output = None

        self.global_state = None
        return

    def reset_variable(self):
        # type : {variable_name : variable_value}
        self.tracked_step_output = {k: {}
                                    for k in self.tracked_step_output.keys()}

        # TODO: [CRITICAL] refactor this
        if self.curr_epoch % self.pertain_n_epoch_output == 0:
            self.run_accumulated_output = []

        self.epoch_accumulated_output = {}

        # * depend on the type of training i.e GAN, the updated accumulated may be different
        self.step_output = None  # holder for output returned after current runstep
        return


####
class RunEngine(object):
    """
    TODO: Include docstring
    """

    def __init__(self,
                 engine_name=None,
                 loader_dict=None,
                 run_step=None,
                 run_info=None,
                 log_info=None,  # TODO: refactor this with trainer.py
                 ):

        self.separate_loader_output = True
        # * auto set all input as object variables
        self.engine_name = engine_name
        self.run_step = run_step

        # * global variable/object holder shared between all event handler
        self.state = State()
        # * check if correctly referenced, not new copies
        self.state.attached_engine_name = engine_name  # TODO: redundant?
        self.state.run_info = run_info
        self.state.log_info = log_info
        self.loader_dict = loader_dict

        # TODO: [CRITICAL] match all the mechanism outline with opt
        self.state.pertain_n_epoch_output = 1 if engine_name == 'valid' else 1

        self.event_handler_dict = {event: [] for event in Events}

        # TODO: think about this more
        # to share global state across a chain of RunEngine such as
        # from the engine for training to engine for validation

        #
        self.terminate = False
        return


    def __reset_state(self):
        # TODO: think about this more, looks too redundant
        new_state = State()
        new_state.attached_engine_name = self.state.attached_engine_name
        new_state.run_info = self.state.run_info
        new_state.log_info = self.state.log_info
        self.state = new_state
        return

    def __trigger_events(self, event):
        for callback in self.event_handler_dict[event]:
            callback.run(self.state, event)
            # TODO: exception and throwing error with name or sthg to allow trace back
        return

    # TODO: variable to indicate output dependency between handler !
    def add_event_handler(self, event_name, handler):
        self.event_handler_dict[event_name].append(handler)

    # ! Put into trainer.py ?
    def run(self, nr_epoch=1, shared_state=None, chained=False):
        def create_pbar(loader):
            pbar_format = 'Processing: |{bar}| {n_fmt}/{total_fmt}[{elapsed}<{remaining},{rate_fmt}]'
            #pbar_format = 'Processing: |{bar}| {n_fmt}/{total_fmt}|{rate_fmt}|'
            if self.engine_name == 'train':
                pbar_format += 'Batch = {postfix[1][Batch]:0.5f}|EMA = {postfix[1][EMA]:0.5f}'
                # * changing print char may break the bar so avoid it
                pbar = tqdm.tqdm(total=len(loader),
                                 leave=True, initial=0,
                                 bar_format=pbar_format, ascii=True,
                                 postfix=['', dict(Batch=float('NaN'),
                                                   EMA=float('NaN'))])
            else:
                pbar = tqdm.tqdm(total=len(loader), leave=True,
                                 bar_format=pbar_format, ascii=True)
            return pbar

        # TODO: refactor this
        if chained:
            self.state.curr_epoch = 0
        self.state.global_state = shared_state

        while self.state.curr_epoch < nr_epoch:
            if not chained:
                logging.info('EPOCH %d' % (self.state.curr_epoch+1))

            # * reset all EMA holder per epoch
            self.state.reset_variable()  
            
            #for loader_name, loader in self.loader_dict.items():
            # * reset all EMA holder, store each loader
            # * data separately and not accumulated
            for key in self.loader_dict.keys():
                if 'PanNuke' in key:
                    loader_name_source = key
                elif 'Lizard' in key and 'few' in key:
                    loader_name_target_few = key
                elif 'Lizard' in key:
                    loader_name_target = key
                else:
                    raise Exception("Check engine!!!")
            loader_source = self.loader_dict[loader_name_source]
            loader_target = self.loader_dict[loader_name_target]
            loader_target_few = self.loader_dict[loader_name_target_few]
                
            if self.separate_loader_output:
                self.state.reset_variable()

            if len(loader_source) > len(loader_target):
                epoch_loader = loader_target
            else:
                epoch_loader = loader_source
              
            self.state.batch_size = epoch_loader.batch_size
            self.__trigger_events(Events.EPOCH_STARTED)
            pbar = create_pbar(epoch_loader)

            if len(loader_source) > len(loader_target):
                raise Exception("Source dataset is larget than the target one!!!")
            else:
                for _iter, data_batch_all in enumerate(zip(loader_source, loader_target, cycle(loader_target_few))):
                    data_batch_source = data_batch_all[0]
                    data_batch_target = data_batch_all[1]
                    data_batch_target_few = data_batch_all[2]
                    self.__trigger_events(Events.STEP_STARTED)

                    step_run_info = [
                        self.state.run_info,
                        {
                            'epoch' : self.state.curr_epoch,
                            'step' : self.state.curr_global_step
                        }
                    ]
                    step_output = self.run_step(data_batch_source, data_batch_target, data_batch_target_few, step_run_info)
                    self.state.step_output = step_output

                    self.__trigger_events(Events.STEP_COMPLETED)
                    self.state.curr_global_step += 1
                    self.state.curr_epoch_step += 1

                    if self.engine_name == 'train':
                        pbar.postfix[1]["Batch"] = step_output['EMA']['overall_loss']
                        pbar.postfix[1]["EMA"] = self.state.tracked_step_output['scalar']['overall_loss']
                    pbar.update()
                    #break
                pbar.close()  # to flush out the bar before doing end of epoch reporting
                if self.separate_loader_output:
                    self.state.curr_epoch += 1
                    self.state.loader_name = loader_name_target
                    self.__trigger_events(Events.EPOCH_COMPLETED)

            if not self.separate_loader_output:
                self.state.curr_epoch += 1
                self.state.loader_name = None
                self.__trigger_events(Events.EPOCH_COMPLETED)

            # TODO: [CRITICAL] align the protocol
            self.state.run_accumulated_output.append(
                self.state.epoch_accumulated_output)

        return
