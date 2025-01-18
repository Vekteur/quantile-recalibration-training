import torch

from ..base_module import BaseModule

from uq.metrics.dist_metrics_computer import DistMetricsComputer, nll
from uq.utils.dist import unnormalize_y
from uq.utils.general import elapsed_timer

class DistModule(BaseModule):
    def compute_metrics(self, dist, y, posthoc_module, batch_idx, monitor_value=None, stage=None, batch=None):
        # We avoid to make costly predictions if we do not compute metrics
        computer = DistMetricsComputer(self)
        if not computer.should_compute(stage):
            return computer.compute_monitor(monitor_value)
        
        # The posthoc module is the identity if there is no posthoc method
        epoch = self.best_epoch_to_use
        if epoch is None:
            epoch = self.current_epoch
        posthoc_module.build(epoch, batch_idx, stage, batch=batch)
        dist = posthoc_module.model(dist)
        # We compute the unnormalized NLL for Figure 1 of the paper
        unnormalized_nll = nll(dist, y).mean()
        if self.rc.config.unnormalize:
            dist = dist.unnormalize(self.scaler)
            y = unnormalize_y(y, self.scaler)
        metrics = DistMetricsComputer(self, y, dist).compute(stage)
        return {'unnormalized_nll': unnormalized_nll, **metrics}
    
    def visualize_computation_graph(self, loss_posthoc):
        import torchviz
        dot = torchviz.make_dot(loss_posthoc, params=dict(self.module.named_parameters()), show_attrs=True, show_saved=True)
        dot.format = 'png'
        dot.render('loss_posthoc')


    def step(self, batch, batch_idx, stage):
        x, y = batch
        y = y.squeeze(dim=-1)
        
        with elapsed_timer() as time:
            pred = self.module.step(x, y, batch_idx, stage)
            loss = self.module.get_loss()
            module_metrics = self.module.get_metrics()
            module_metrics['loss'] = loss
            es_loss = module_metrics['es_loss']
        self.advance_timer('step_time', time())

        with elapsed_timer() as time:
            if self.rc.config.plot_toy and self.current_epoch % 100 == 0 and batch_idx == 0:
                from uq.analysis.plot_toy import plot_inhoc
                plot_inhoc(self, x, y, batch_idx, stage)

            if self.rc.config.plot_predictions and self.current_epoch % 5 == 0 and batch_idx == 0:
                from uq.analysis.plot_predictions import plot_predictions_per_epoch
                plot_predictions_per_epoch(self, batch_idx, stage)
                from uq.analysis.plot_predictions import plot_inhoc
                plot_inhoc(self, x, y, batch_idx, stage)

            posthoc_metrics_list = []
            for posthoc_module in self.posthoc_manager.modules:
                with elapsed_timer() as time:
                    with torch.no_grad():
                        metrics = self.compute_metrics(pred, y, posthoc_module, batch_idx, monitor_value=es_loss.detach(), stage=stage, batch=batch)
                    posthoc_metrics = {**metrics, **module_metrics}
                    posthoc_metrics_list.append(posthoc_metrics)
                posthoc_method = posthoc_module.hparams['method']
                self.advance_timer(f'metrics_{posthoc_method}_time', time())
        for started_stage in self.started_stages:
            self.advance_timer('time', time(), cancel_time=True, stage=started_stage)

        if stage == 'train':
            with elapsed_timer() as time:
                opt = self.optimizers()
                opt.zero_grad()
                self.manual_backward(loss)
            self.advance_timer('backward_time', time())
            with elapsed_timer() as time:
                opt.step()
            self.advance_timer('opt_step_time', time())

        # `loss` is needed for the optimization and `es_loss` for early stopping
        return {'loss': loss, 'es_loss': es_loss, 'posthoc_metrics_list': posthoc_metrics_list}
