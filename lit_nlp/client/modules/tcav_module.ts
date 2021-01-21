/**
 * @license
 * Copyright 2020 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// tslint:disable:no-new-decorators
import '../elements/spinner';
import '../elements/bar_chart_vis';

import {customElement, html} from 'lit-element';
import {computed, observable} from 'mobx';

import {app} from '../core/lit_app';
import {LitModule} from '../core/lit_module';
import {CallConfig, ModelInfoMap, Spec} from '../lib/types';
import {doesOutputSpecContain, findSpecKeys} from '../lib/utils';
// import {GroupService} from '../services/group_service';
import {SliceService} from '../services/services';

import {styles as sharedStyles} from './shared_styles.css';
import {styles} from './tcav_module.css';

const MIN_EXAMPLES_LENGTH = 2;  // minimum examples needed to train the LM.
const ALL = 'all';
const TCAV_NAME = 'tcav';
const CHART_MARGIN = 30;
const CHART_WIDTH = 150;
const CHART_HEIGHT = 150;

/**
 * The TCAV module.
 */
@customElement('tcav-module')
export class TCAVModule extends LitModule {
  static get styles() {
    return [sharedStyles, styles];
  }
  static title = 'TCAV';
  static numCols = 3;
  static duplicateForModelComparison = true;

  static template = (model = '') => {
    return html`
      <tcav-module model=${model}>
      </tcav-module>`;
  };
  private readonly sliceService = app.getService(SliceService);

  private scores = new Map<string, number>();
  @observable private selectedSlice: string = ALL;
  @observable private selectedLayer: string = '';
  @observable private selectedClass: string = '';
  @observable private isLoading: boolean = false;

  @computed
  get modelSpec() {
    return this.appState.getModelSpec(this.model);
  }

  @computed
  get gradKeys() {
    return findSpecKeys(this.modelSpec.output, 'Gradients');
  }

  @computed
  get predClasses() {
    const predKeys = findSpecKeys(this.modelSpec.output, 'MulticlassPreds');
    // TODO(@lit-dev): Handle the multi-headed case with more than one pred key.
    return this.modelSpec.output[predKeys[0]].vocab!;
  }

  firstUpdated() {
    // Set the first grad key as default in selector.
    if (this.selectedLayer === '' && this.gradKeys.length > 0) {
      this.selectedLayer = this.gradKeys[0];
    }
    // Set first pred class as default in selector.
    if (this.selectedClass === '' && this.predClasses.length > 0) {
      this.selectedClass = this.predClasses[0];
    }
  }

  renderSelector(
      fieldName: string, handleChange: (e: Event) => void, selected: string,
      items: string[]) {
    // clang-format off
    return html `
        <div class='field-name'>${fieldName}</div>
        <select class="dropdown" @change=${handleChange}>
          <option value="${ALL}">${ALL}</option>
          ${items.map(val => {
              return html`
                  <option value="${val}"
                  ?selected=${val === selected}>${val}</option>
                  `;
          })}
        </select>
        `;
    // clang-format on
  }

  renderSelectors() {
    const handleSliceChange = (e: Event) => {
      const selected = e.target as HTMLInputElement;
      this.selectedSlice = selected.value;
    };
    const handleLayerChange = (e: Event) => {
      const selected = e.target as HTMLInputElement;
      this.selectedLayer = selected.value;
    };
    const handleClassChange = (e: Event) => {
      const selected = e.target as HTMLInputElement;
      this.selectedClass = selected.value;
    };

    // clang-format off
    return html`
        ${this.renderSelector('Slice', handleSliceChange, this.selectedSlice,
                              this.sliceService.sliceNames)}
        ${this.renderSelector('Layer/Bottleneck', handleLayerChange,
                              this.selectedLayer, this.gradKeys)}
        ${this.renderSelector('Class to Explain', handleClassChange,
                              this.selectedClass, this.predClasses)}
        `;
    // clang-format on
  }

  render() {
    const shouldDisable = () => {
      const slices = (this.selectedSlice === ALL) ?
          this.sliceService.sliceNames :
          [this.selectedSlice];
      for (const slice of slices) {
        const examples = this.sliceService.getSliceByName(slice);
        if (examples == null) return true;
        if (examples.length >= MIN_EXAMPLES_LENGTH) {
          return false;  // only enable if slice has minimum number of examples
        }
      }
      return true;
    };
    // clang-format off
    return html`
      <div id="outer-container">
        <div class="container">
          ${this.renderSelectors()}
          <button id='submit' @click=${() => this.runTCAV()} ?disabled=${
        shouldDisable()}
          >Run TCAV</button>
        </div>
        <bar-chart-vis
          .scores=${this.scores}
          .isLoading=${this.isLoading}
          .width=${CHART_WIDTH}
          .height=${CHART_HEIGHT}
          .margin=${CHART_MARGIN}
        ></bar-chart-vis>
      </div>
    `;
    // clang-format on
  }

  private async runTCAV() {
    this.isLoading = true;
    const scores = new Map<string, number>();

    // TODO(litdev): Add option to run TCAV on selected examples.
    // Run TCAV for all slices if 'all' is selected.
    const slicesToRun = this.selectedSlice === ALL ?
        this.sliceService.sliceNames :
        [this.selectedSlice];
    for (const slice of slicesToRun) {
      const selectedIds = this.sliceService.getSliceByName(slice);
      if (selectedIds == null || selectedIds.length < MIN_EXAMPLES_LENGTH) {
        continue;
      }

      const config = {
        'concept_set_ids': selectedIds,
        'class_to_explain': this.selectedClass,
        'grad_layer': this.selectedLayer,
      } as CallConfig;

      // All indexed inputs in the dataset are passed in, with the concept set
      // ids specified in the config.
      const promise = this.apiService.getInterpretations(
          this.appState.currentInputData, this.model,
          this.appState.currentDataset, TCAV_NAME, config, `Running ${name}`);
      const result =
          await this.loadLatest(`interpretations-${TCAV_NAME}`, promise);
      if (result === null) {
        this.isLoading = false;
        return;
      }
      // Only shows examples with a p-value less than 0.05.
      // TODO(lit-dev): Add display text when the concepts have high p-values.
      // and are discarded.
      if (result[0]['p_val'] < 0.5) {
        const score = result[0]['result']['score'];
        const axisLabel = slice;
        scores.set(axisLabel, score);
      }
    }
    this.isLoading = false;
    this.scores = scores;
  }

  static shouldDisplayModule(modelSpecs: ModelInfoMap, datasetSpec: Spec) {
    const supportsEmbs = doesOutputSpecContain(modelSpecs, 'Embeddings');
    const supportsGrads = doesOutputSpecContain(modelSpecs, 'Gradients');
    const multiclassPreds =
        doesOutputSpecContain(modelSpecs, 'MulticlassPreds');
    if (supportsGrads && supportsEmbs && multiclassPreds) {
      return true;
    }
    return false;
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'tcav-module': TCAVModule;
  }
}
