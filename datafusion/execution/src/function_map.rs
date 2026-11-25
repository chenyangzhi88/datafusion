// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

/// Published function names with context-local replacements and removals.
/// A point lookup only examines the current changes and the published map.
#[derive(Debug)]
pub struct FunctionMap<F> {
    base: Arc<HashMap<String, Arc<F>>>,
    changes: Arc<HashMap<String, Option<Arc<F>>>>,
    merged: Arc<OnceLock<Arc<HashMap<String, Arc<F>>>>>,
}

impl<F> Default for FunctionMap<F> {
    fn default() -> Self {
        Self::from(HashMap::new())
    }
}

impl<F> Clone for FunctionMap<F> {
    fn clone(&self) -> Self {
        Self {
            base: Arc::clone(&self.base),
            changes: Arc::clone(&self.changes),
            merged: Arc::clone(&self.merged),
        }
    }
}

impl<F> From<HashMap<String, Arc<F>>> for FunctionMap<F> {
    fn from(base: HashMap<String, Arc<F>>) -> Self {
        Self {
            base: Arc::new(base),
            changes: Arc::default(),
            merged: Arc::default(),
        }
    }
}

impl<F> FunctionMap<F> {
    pub fn get(&self, name: &str) -> Option<&Arc<F>> {
        match self.changes.get(name) {
            Some(value) => value.as_ref(),
            None => self.base.get(name),
        }
    }

    pub fn names(&self) -> impl Iterator<Item = &String> {
        self.entries().map(|(name, _)| name)
    }

    fn entries(&self) -> impl Iterator<Item = (&String, &Arc<F>)> {
        self.base
            .iter()
            .filter(move |(name, _)| !self.changes.contains_key(*name))
            .chain(
                self.changes.iter().filter_map(|(name, value)| {
                    value.as_ref().map(|value| (name, value))
                }),
            )
    }

    pub fn insert(&mut self, name: String, value: Arc<F>) -> Option<Arc<F>> {
        let previous = self.get(&name).cloned();
        self.changes_mut().insert(name, Some(value));
        previous
    }

    pub fn remove(&mut self, name: &str) -> Option<Arc<F>> {
        let previous = self.get(name).cloned()?;
        if self.base.contains_key(name) {
            self.changes_mut().insert(name.to_owned(), None);
        } else {
            self.changes_mut().remove(name);
        }
        Some(previous)
    }

    pub fn as_map(&self) -> &HashMap<String, Arc<F>> {
        self.merged().as_ref()
    }

    pub fn publish(&mut self) {
        if !self.changes.is_empty() {
            self.base = Arc::clone(self.merged());
            self.changes = Arc::default();
            self.merged = Arc::default();
        }
    }

    fn merged(&self) -> &Arc<HashMap<String, Arc<F>>> {
        if self.changes.is_empty() {
            return &self.base;
        }
        self.merged.get_or_init(|| {
            Arc::new(
                self.entries()
                    .map(|(name, value)| (name.clone(), Arc::clone(value)))
                    .collect(),
            )
        })
    }

    fn changes_mut(&mut self) -> &mut HashMap<String, Option<Arc<F>>> {
        if let Some(merged) = Arc::get_mut(&mut self.merged) {
            let _ = merged.take();
        } else {
            self.merged = Arc::default();
        }
        Arc::make_mut(&mut self.changes)
    }
}
