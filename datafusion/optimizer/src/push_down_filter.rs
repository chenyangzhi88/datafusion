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

//! [`PushDownFilter`] applies filters as early as possible

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use arrow::datatypes::DataType;
use indexmap::IndexSet;
use itertools::Itertools;

use datafusion_common::tree_node::{
    Transformed, TransformedResult, TreeNode, TreeNodeRecursion,
};
use datafusion_common::{
    Column, DFSchema, Result, assert_eq_or_internal_err, assert_or_internal_err,
    internal_err, plan_err, qualified_name,
};
use datafusion_expr::expr::WindowFunction;
use datafusion_expr::expr_rewriter::replace_col;
use datafusion_expr::logical_plan::{Join, JoinType, LogicalPlan, TableScan, Union};
use datafusion_expr::utils::{
    conjunction, expr_to_columns, split_conjunction, split_conjunction_owned,
};
use datafusion_expr::{
    BinaryExpr, Expr, Filter, Operator, Projection, TableProviderFilterPushDown, and, or,
};

use crate::optimizer::ApplyOrder;
use crate::simplify_expressions::simplify_predicates;
use crate::utils::{has_all_column_refs, is_restrict_null_predicate};
use crate::{OptimizerConfig, OptimizerRule};
use datafusion_expr::ExpressionPlacement;

/// Optimizer rule for pushing (moving) filter expressions down in a plan so
/// they are applied as early as possible.
///
/// # Introduction
///
/// The goal of this rule is to improve query performance by eliminating
/// redundant work.
///
/// For example, given a plan that sorts all values where `a > 10`:
///
/// ```text
///  Filter (a > 10)
///    Sort (a, b)
/// ```
///
/// A better plan is to  filter the data *before* the Sort, which sorts fewer
/// rows and therefore does less work overall:
///
/// ```text
///  Sort (a, b)
///    Filter (a > 10)  <-- Filter is moved before the sort
/// ```
///
/// However it is not always possible to push filters down. For example, given a
/// plan that finds the top 3 values and then keeps only those that are greater
/// than 10, if the filter is pushed below the limit it would produce a
/// different result.
///
/// ```text
///  Filter (a > 10)   <-- can not move this Filter before the limit
///    Limit (fetch=3)
///      Sort (a, b)
/// ```
///
///
/// More formally, a filter-commutative operation is an operation `op` that
/// satisfies `filter(op(data)) = op(filter(data))`.
///
/// The filter-commutative property is plan and column-specific. A filter on `a`
/// can be pushed through a `Aggregate(group_by = [a], agg=[sum(b))`. However, a
/// filter on  `sum(b)` can not be pushed through the same aggregate.
///
/// # Handling Conjunctions
///
/// It is possible to only push down **part** of a filter expression if is
/// connected with `AND`s (more formally if it is a "conjunction").
///
/// For example, given the following plan:
///
/// ```text
/// Filter(a > 10 AND sum(b) < 5)
///   Aggregate(group_by = [a], agg = [sum(b))
/// ```
///
/// The `a > 10` is commutative with the `Aggregate` but  `sum(b) < 5` is not.
/// Therefore it is possible to only push part of the expression, resulting in:
///
/// ```text
/// Filter(sum(b) < 5)
///   Aggregate(group_by = [a], agg = [sum(b))
///     Filter(a > 10)
/// ```
///
/// # Handling Column Aliases
///
/// This optimizer must sometimes handle re-writing filter expressions when they
/// pushed, for example if there is a projection that aliases `a+1` to `"b"`:
///
/// ```text
/// Filter (b > 10)
///     Projection: [a+1 AS "b"]  <-- changes the name of `a+1` to `b`
/// ```
///
/// To apply the filter prior to the `Projection`, all references to `b` must be
/// rewritten to `a+1`:
///
/// ```text
/// Projection: a AS "b"
///     Filter: (a + 1 > 10)  <--- changed from b to a + 1
/// ```
/// # Implementation Notes
///
/// This implementation performs a single pass through the plan, "pushing" down
/// filters. When it passes through a filter, it stores that filter, and when it
/// reaches a plan node that does not commute with that filter, it adds the
/// filter to that place. When it passes through a projection, it re-writes the
/// filter's expression taking into account that projection.
#[derive(Default, Debug)]
pub struct PushDownFilter {}

/// For a given JOIN type, determine whether each input of the join is preserved
/// for post-join (`WHERE` clause) filters.
///
/// It is only correct to push filters below a join for preserved inputs.
///
/// # Return Value
/// A tuple of booleans - (left_preserved, right_preserved).
///
/// # "Preserved" input definition
///
/// We say a join side is preserved if the join returns all or a subset of the rows from
/// the relevant side, such that each row of the output table directly maps to a row of
/// the preserved input table. If a table is not preserved, it can provide extra null rows.
/// That is, there may be rows in the output table that don't directly map to a row in the
/// input table.
///
/// For example:
///   - In an inner join, both sides are preserved, because each row of the output
///     maps directly to a row from each side.
///
///   - In a left join, the left side is preserved (we can push predicates) but
///     the right is not, because there may be rows in the output that don't
///     directly map to a row in the right input (due to nulls filling where there
///     is no match on the right).
pub(crate) fn lr_is_preserved(join_type: JoinType) -> (bool, bool) {
    match join_type {
        JoinType::Inner => (true, true),
        JoinType::Left => (true, false),
        JoinType::Right => (false, true),
        JoinType::Full => (false, false),
        // No columns from the right side of the join can be referenced in output
        // predicates for semi/anti joins, so whether we specify t/f doesn't matter.
        JoinType::LeftSemi | JoinType::LeftAnti | JoinType::LeftMark => (true, false),
        // No columns from the left side of the join can be referenced in output
        // predicates for semi/anti joins, so whether we specify t/f doesn't matter.
        JoinType::RightSemi | JoinType::RightAnti | JoinType::RightMark => (false, true),
    }
}

/// For a given JOIN type, determine whether each input of the join is preserved
/// for the join condition (`ON` clause filters).
///
/// It is only correct to push filters below a join for preserved inputs.
///
/// # Return Value
/// A tuple of booleans - (left_preserved, right_preserved).
///
/// See [`lr_is_preserved`] for a definition of "preserved".
pub(crate) fn on_lr_is_preserved(join_type: JoinType) -> (bool, bool) {
    match join_type {
        JoinType::Inner => (true, true),
        JoinType::Left => (false, true),
        JoinType::Right => (true, false),
        JoinType::Full => (false, false),
        JoinType::LeftSemi | JoinType::RightSemi => (true, true),
        JoinType::LeftAnti => (false, true),
        JoinType::RightAnti => (true, false),
        JoinType::LeftMark => (false, true),
        JoinType::RightMark => (true, false),
    }
}

/// Evaluates the columns referenced in the given expression to see if they refer
/// only to the left or right columns
#[derive(Debug)]
struct ColumnChecker<'a> {
    /// schema of left join input
    left_schema: &'a DFSchema,
    /// columns in left_schema, computed on demand
    left_columns: Option<HashSet<Column>>,
    /// schema of right join input
    right_schema: &'a DFSchema,
    /// columns in left_schema, computed on demand
    right_columns: Option<HashSet<Column>>,
}

impl<'a> ColumnChecker<'a> {
    fn new(left_schema: &'a DFSchema, right_schema: &'a DFSchema) -> Self {
        Self {
            left_schema,
            left_columns: None,
            right_schema,
            right_columns: None,
        }
    }

    /// Return true if the expression references only columns from the left side of the join
    fn is_left_only(&mut self, predicate: &Expr) -> bool {
        if self.left_columns.is_none() {
            self.left_columns = Some(schema_columns(self.left_schema));
        }
        has_all_column_refs(predicate, self.left_columns.as_ref().unwrap())
    }

    /// Return true if the expression references only columns from the right side of the join
    fn is_right_only(&mut self, predicate: &Expr) -> bool {
        if self.right_columns.is_none() {
            self.right_columns = Some(schema_columns(self.right_schema));
        }
        has_all_column_refs(predicate, self.right_columns.as_ref().unwrap())
    }
}

/// Returns all columns in the schema
fn schema_columns(schema: &DFSchema) -> HashSet<Column> {
    schema
        .iter()
        .flat_map(|(qualifier, field)| {
            [
                Column::new(qualifier.cloned(), field.name()),
                // we need to push down filter using unqualified column as well
                Column::new_unqualified(field.name()),
            ]
        })
        .collect::<HashSet<_>>()
}

/// Determine whether the predicate can evaluate as the join conditions
fn can_evaluate_as_join_condition(predicate: &Expr) -> Result<bool> {
    let mut is_evaluate = true;
    predicate.apply(|expr| match expr {
        Expr::Column(_)
        | Expr::Literal(_, _)
        | Expr::Placeholder(_)
        | Expr::ScalarVariable(_, _) => Ok(TreeNodeRecursion::Jump),
        Expr::Exists { .. }
        | Expr::InSubquery(_)
        | Expr::SetComparison(_)
        | Expr::ScalarSubquery(_)
        | Expr::OuterReferenceColumn(_, _)
        | Expr::Unnest(_) => {
            is_evaluate = false;
            Ok(TreeNodeRecursion::Stop)
        }
        Expr::Alias(_)
        | Expr::BinaryExpr(_)
        | Expr::Like(_)
        | Expr::SimilarTo(_)
        | Expr::Not(_)
        | Expr::IsNotNull(_)
        | Expr::IsNull(_)
        | Expr::IsTrue(_)
        | Expr::IsFalse(_)
        | Expr::IsUnknown(_)
        | Expr::IsNotTrue(_)
        | Expr::IsNotFalse(_)
        | Expr::IsNotUnknown(_)
        | Expr::Negative(_)
        | Expr::Between(_)
        | Expr::Case(_)
        | Expr::Cast(_)
        | Expr::TryCast(_)
        | Expr::InList { .. }
        | Expr::ScalarFunction(_) => Ok(TreeNodeRecursion::Continue),
        // TODO: remove the next line after `Expr::Wildcard` is removed
        #[expect(deprecated)]
        Expr::AggregateFunction(_)
        | Expr::WindowFunction(_)
        | Expr::Wildcard { .. }
        | Expr::GroupingSet(_) => internal_err!("Unsupported predicate type"),
    })?;
    Ok(is_evaluate)
}

/// examine OR clause to see if any useful clauses can be extracted and push down.
/// extract at least one qual from each sub clauses of OR clause, then form the quals
/// to new OR clause as predicate.
///
/// # Example
/// ```text
/// Filter: (a = c and a < 20) or (b = d and b > 10)
///     join/crossjoin:
///          TableScan: projection=[a, b]
///          TableScan: projection=[c, d]
/// ```
///
/// is optimized to
///
/// ```text
/// Filter: (a = c and a < 20) or (b = d and b > 10)
///     join/crossjoin:
///          Filter: (a < 20) or (b > 10)
///              TableScan: projection=[a, b]
///          TableScan: projection=[c, d]
/// ```
///
/// In general, predicates of this form:
///
/// ```sql
/// (A AND B) OR (C AND D)
/// ```
///
/// will be transformed to one of:
///
/// * `((A AND B) OR (C AND D)) AND (A OR C)`
/// * `((A AND B) OR (C AND D)) AND ((A AND B) OR C)`
/// * do nothing.
fn extract_or_clauses_for_join<'a>(
    filters: &'a [Expr],
    schema: &'a DFSchema,
) -> impl Iterator<Item = Expr> + 'a {
    let schema_columns = schema_columns(schema);

    // new formed OR clauses and their column references
    filters.iter().filter_map(move |expr| {
        if let Expr::BinaryExpr(BinaryExpr {
            left,
            op: Operator::Or,
            right,
        }) = expr
        {
            let left_expr = extract_or_clause(left.as_ref(), &schema_columns);
            let right_expr = extract_or_clause(right.as_ref(), &schema_columns);

            // If nothing can be extracted from any sub clauses, do nothing for this OR clause.
            if let (Some(left_expr), Some(right_expr)) = (left_expr, right_expr) {
                return Some(or(left_expr, right_expr));
            }
        }
        None
    })
}

/// extract qual from OR sub-clause.
///
/// A qual is extracted if it only contains set of column references in schema_columns.
///
/// For AND clause, we extract from both sub-clauses, then make new AND clause by extracted
/// clauses if both extracted; Otherwise, use the extracted clause from any sub-clauses or None.
///
/// For OR clause, we extract from both sub-clauses, then make new OR clause by extracted clauses if both extracted;
/// Otherwise, return None.
///
/// For other clause, apply the rule above to extract clause.
fn extract_or_clause(expr: &Expr, schema_columns: &HashSet<Column>) -> Option<Expr> {
    let mut predicate = None;

    match expr {
        Expr::BinaryExpr(BinaryExpr {
            left: l_expr,
            op: Operator::Or,
            right: r_expr,
        }) => {
            let l_expr = extract_or_clause(l_expr, schema_columns);
            let r_expr = extract_or_clause(r_expr, schema_columns);

            if let (Some(l_expr), Some(r_expr)) = (l_expr, r_expr) {
                predicate = Some(or(l_expr, r_expr));
            }
        }
        Expr::BinaryExpr(BinaryExpr {
            left: l_expr,
            op: Operator::And,
            right: r_expr,
        }) => {
            let l_expr = extract_or_clause(l_expr, schema_columns);
            let r_expr = extract_or_clause(r_expr, schema_columns);

            match (l_expr, r_expr) {
                (Some(l_expr), Some(r_expr)) => {
                    predicate = Some(and(l_expr, r_expr));
                }
                (Some(l_expr), None) => {
                    predicate = Some(l_expr);
                }
                (None, Some(r_expr)) => {
                    predicate = Some(r_expr);
                }
                (None, None) => {
                    predicate = None;
                }
            }
        }
        _ => {
            if has_all_column_refs(expr, schema_columns) {
                predicate = Some(expr.clone());
            }
        }
    }

    predicate
}

/// push down join/cross-join
fn push_down_all_join(
    predicates: Vec<Expr>,
    inferred_join_predicates: Vec<Expr>,
    mut join: Join,
    on_filter: Vec<Expr>,
) -> Result<Transformed<LogicalPlan>> {
    let is_inner_join = join.join_type == JoinType::Inner;
    // Get pushable predicates from current optimizer state
    let (left_preserved, right_preserved) = lr_is_preserved(join.join_type);

    // The predicates can be divided to three categories:
    // 1) can push through join to its children(left or right)
    // 2) can be converted to join conditions if the join type is Inner
    // 3) should be kept as filter conditions
    let left_schema = join.left.schema();
    let right_schema = join.right.schema();
    let mut left_push = vec![];
    let mut right_push = vec![];
    let mut keep_predicates = vec![];
    let mut join_conditions = vec![];
    let mut checker = ColumnChecker::new(left_schema, right_schema);
    for predicate in predicates {
        if left_preserved && checker.is_left_only(&predicate) {
            left_push.push(predicate);
        } else if right_preserved && checker.is_right_only(&predicate) {
            right_push.push(predicate);
        } else if is_inner_join && can_evaluate_as_join_condition(&predicate)? {
            // Here we do not differ it is eq or non-eq predicate, ExtractEquijoinPredicate will extract the eq predicate
            // and convert to the join on condition
            join_conditions.push(predicate);
        } else {
            keep_predicates.push(predicate);
        }
    }

    // Push predicates inferred from the join expression
    for predicate in inferred_join_predicates {
        if checker.is_left_only(&predicate) {
            left_push.push(predicate);
        } else if checker.is_right_only(&predicate) {
            right_push.push(predicate);
        }
    }

    let mut on_filter_join_conditions = vec![];
    let (on_left_preserved, on_right_preserved) = on_lr_is_preserved(join.join_type);

    if !on_filter.is_empty() {
        for on in on_filter {
            if on_left_preserved && checker.is_left_only(&on) {
                left_push.push(on)
            } else if on_right_preserved && checker.is_right_only(&on) {
                right_push.push(on)
            } else {
                on_filter_join_conditions.push(on)
            }
        }
    }

    // Extract from OR clause, generate new predicates for both side of join if possible.
    // We only track the unpushable predicates above.
    if left_preserved {
        left_push.extend(extract_or_clauses_for_join(&keep_predicates, left_schema));
        left_push.extend(extract_or_clauses_for_join(&join_conditions, left_schema));
    }
    if right_preserved {
        right_push.extend(extract_or_clauses_for_join(&keep_predicates, right_schema));
        right_push.extend(extract_or_clauses_for_join(&join_conditions, right_schema));
    }

    // For predicates from join filter, we should check with if a join side is preserved
    // in term of join filtering.
    if on_left_preserved {
        left_push.extend(extract_or_clauses_for_join(
            &on_filter_join_conditions,
            left_schema,
        ));
    }
    if on_right_preserved {
        right_push.extend(extract_or_clauses_for_join(
            &on_filter_join_conditions,
            right_schema,
        ));
    }

    if let Some(predicate) = conjunction(left_push) {
        join.left = Arc::new(LogicalPlan::Filter(Filter::try_new(predicate, join.left)?));
    }
    if let Some(predicate) = conjunction(right_push) {
        join.right =
            Arc::new(LogicalPlan::Filter(Filter::try_new(predicate, join.right)?));
    }

    // Add any new join conditions as the non join predicates
    join_conditions.extend(on_filter_join_conditions);
    join.filter = conjunction(join_conditions);

    // wrap the join on the filter whose predicates must be kept, if any
    let plan = LogicalPlan::Join(join);
    let plan = if let Some(predicate) = conjunction(keep_predicates) {
        LogicalPlan::Filter(Filter::try_new(predicate, Arc::new(plan))?)
    } else {
        plan
    };
    Ok(Transformed::yes(plan))
}

fn push_down_join(
    join: Join,
    parent_predicate: Option<&Expr>,
) -> Result<Transformed<LogicalPlan>> {
    // Split the parent predicate into individual conjunctive parts.
    let predicates = parent_predicate
        .map_or_else(Vec::new, |pred| split_conjunction_owned(pred.clone()));

    // Extract conjunctions from the JOIN's ON filter, if present.
    let on_filters = join
        .filter
        .as_ref()
        .map_or_else(Vec::new, |filter| split_conjunction_owned(filter.clone()));

    // Are there any new join predicates that can be inferred from the filter expressions?
    let inferred_join_predicates =
        infer_join_predicates(&join, &predicates, &on_filters)?;

    if on_filters.is_empty()
        && predicates.is_empty()
        && inferred_join_predicates.is_empty()
    {
        return Ok(Transformed::no(LogicalPlan::Join(join)));
    }

    push_down_all_join(predicates, inferred_join_predicates, join, on_filters)
}

/// Extracts any equi-join join predicates from the given filter expressions.
///
/// Parameters
/// * `join` the join in question
///
/// * `predicates` the pushed down filter expression
///
/// * `on_filters` filters from the join ON clause that have not already been
///   identified as join predicates
fn infer_join_predicates(
    join: &Join,
    predicates: &[Expr],
    on_filters: &[Expr],
) -> Result<Vec<Expr>> {
    // Only allow both side key is column.
    let join_col_keys = join
        .on
        .iter()
        .filter_map(|(l, r)| {
            let left_col = l.try_as_col()?;
            let right_col = r.try_as_col()?;
            Some((left_col, right_col))
        })
        .collect::<Vec<_>>();

    let join_type = join.join_type;

    let mut inferred_predicates = InferredPredicates::new(join_type);

    infer_join_predicates_from_predicates(
        &join_col_keys,
        predicates,
        &mut inferred_predicates,
    )?;

    infer_join_predicates_from_on_filters(
        &join_col_keys,
        join_type,
        on_filters,
        &mut inferred_predicates,
    )?;

    Ok(inferred_predicates.predicates)
}

/// Inferred predicates collector.
/// When the JoinType is not Inner, we need to detect whether the inferred predicate can strictly
/// filter out NULL, otherwise ignore it. e.g.
/// ```text
/// SELECT * FROM t1 LEFT JOIN t2 ON t1.c0 = t2.c0 WHERE t2.c0 IS NULL;
/// ```
/// We cannot infer the predicate `t1.c0 IS NULL`, otherwise the predicate will be pushed down to
/// the left side, resulting in the wrong result.
struct InferredPredicates {
    predicates: Vec<Expr>,
    is_inner_join: bool,
}

impl InferredPredicates {
    fn new(join_type: JoinType) -> Self {
        Self {
            predicates: vec![],
            is_inner_join: join_type == JoinType::Inner,
        }
    }

    fn try_build_predicate(
        &mut self,
        predicate: Expr,
        replace_map: &HashMap<&Column, &Column>,
    ) -> Result<()> {
        if self.is_inner_join
            || matches!(
                is_restrict_null_predicate(
                    predicate.clone(),
                    replace_map.keys().cloned()
                ),
                Ok(true)
            )
        {
            self.predicates.push(replace_col(predicate, replace_map)?);
        }

        Ok(())
    }
}

/// Infer predicates from the pushed down predicates.
///
/// Parameters
/// * `join_col_keys` column pairs from the join ON clause
///
/// * `predicates` the pushed down predicates
///
/// * `inferred_predicates` the inferred results
fn infer_join_predicates_from_predicates(
    join_col_keys: &[(&Column, &Column)],
    predicates: &[Expr],
    inferred_predicates: &mut InferredPredicates,
) -> Result<()> {
    infer_join_predicates_impl::<true, true>(
        join_col_keys,
        predicates,
        inferred_predicates,
    )
}

/// Infer predicates from the join filter.
///
/// Parameters
/// * `join_col_keys` column pairs from the join ON clause
///
/// * `join_type` the JoinType of Join
///
/// * `on_filters` filters from the join ON clause that have not already been
///   identified as join predicates
///
/// * `inferred_predicates` the inferred results
fn infer_join_predicates_from_on_filters(
    join_col_keys: &[(&Column, &Column)],
    join_type: JoinType,
    on_filters: &[Expr],
    inferred_predicates: &mut InferredPredicates,
) -> Result<()> {
    match join_type {
        JoinType::Full | JoinType::LeftAnti | JoinType::RightAnti => Ok(()),
        JoinType::Inner => infer_join_predicates_impl::<true, true>(
            join_col_keys,
            on_filters,
            inferred_predicates,
        ),
        JoinType::Left | JoinType::LeftSemi | JoinType::LeftMark => {
            infer_join_predicates_impl::<true, false>(
                join_col_keys,
                on_filters,
                inferred_predicates,
            )
        }
        JoinType::Right | JoinType::RightSemi | JoinType::RightMark => {
            infer_join_predicates_impl::<false, true>(
                join_col_keys,
                on_filters,
                inferred_predicates,
            )
        }
    }
}

/// Infer predicates from the given predicates.
///
/// Parameters
/// * `join_col_keys` column pairs from the join ON clause
///
/// * `input_predicates` the given predicates. It can be the pushed down predicates,
///   or it can be the filters of the Join
///
/// * `inferred_predicates` the inferred results
///
/// * `ENABLE_LEFT_TO_RIGHT` indicates that the right table related predicate can
///   be inferred from the left table related predicate
///
/// * `ENABLE_RIGHT_TO_LEFT` indicates that the left table related predicate can
///   be inferred from the right table related predicate
fn infer_join_predicates_impl<
    const ENABLE_LEFT_TO_RIGHT: bool,
    const ENABLE_RIGHT_TO_LEFT: bool,
>(
    join_col_keys: &[(&Column, &Column)],
    input_predicates: &[Expr],
    inferred_predicates: &mut InferredPredicates,
) -> Result<()> {
    for predicate in input_predicates {
        let mut join_cols_to_replace = HashMap::new();

        for &col in &predicate.column_refs() {
            for (l, r) in join_col_keys.iter() {
                if ENABLE_LEFT_TO_RIGHT && col == *l {
                    join_cols_to_replace.insert(col, *r);
                    break;
                }
                if ENABLE_RIGHT_TO_LEFT && col == *r {
                    join_cols_to_replace.insert(col, *l);
                    break;
                }
            }
        }
        if join_cols_to_replace.is_empty() {
            continue;
        }

        inferred_predicates
            .try_build_predicate(predicate.clone(), &join_cols_to_replace)?;
    }
    Ok(())
}

impl OptimizerRule for PushDownFilter {
    fn name(&self) -> &str {
        "push_down_filter"
    }

    fn apply_order(&self) -> Option<ApplyOrder> {
        Some(ApplyOrder::TopDown)
    }

    fn supports_rewrite(&self) -> bool {
        true
    }

    fn rewrite(
        &self,
        plan: LogicalPlan,
        config: &dyn OptimizerConfig,
    ) -> Result<Transformed<LogicalPlan>> {
        let _ = config.options();
        if let LogicalPlan::Join(join) = plan {
            return push_down_join(join, None);
        };

        let plan_schema = Arc::clone(plan.schema());

        let LogicalPlan::Filter(mut filter) = plan else {
            return Ok(Transformed::no(plan));
        };

        let predicate = split_conjunction_owned(filter.predicate.clone());
        let old_predicate_len = predicate.len();
        let new_predicates = simplify_predicates(predicate)?;
        if old_predicate_len != new_predicates.len() {
            let Some(new_predicate) = conjunction(new_predicates) else {
                // new_predicates is empty - remove the filter entirely
                // Return the child plan without the filter
                return Ok(Transformed::yes(Arc::unwrap_or_clone(filter.input)));
            };
            filter.predicate = new_predicate;
        }

        // If the child has a fetch (limit) or skip (offset), pushing a filter
        // below it would change semantics: the limit/offset should apply before
        // the filter, not after.
        if filter.input.fetch()?.is_some() || filter.input.skip()?.is_some() {
            return Ok(Transformed::no(LogicalPlan::Filter(filter)));
        }

        match Arc::unwrap_or_clone(filter.input) {
            LogicalPlan::Filter(child_filter) => {
                let parents_predicates = split_conjunction_owned(filter.predicate);

                // remove duplicated filters
                let child_predicates = split_conjunction_owned(child_filter.predicate);
                let new_predicates = parents_predicates
                    .into_iter()
                    .chain(child_predicates)
                    // use IndexSet to remove dupes while preserving predicate order
                    .collect::<IndexSet<_>>()
                    .into_iter()
                    .collect::<Vec<_>>();

                let Some(new_predicate) = conjunction(new_predicates) else {
                    return plan_err!("at least one expression exists");
                };
                let new_filter = LogicalPlan::Filter(Filter::try_new(
                    new_predicate,
                    child_filter.input,
                )?);
                self.rewrite(new_filter, config)
            }
            LogicalPlan::Repartition(repartition) => {
                let new_filter =
                    Filter::try_new(filter.predicate, Arc::clone(&repartition.input))
                        .map(LogicalPlan::Filter)?;
                insert_below(LogicalPlan::Repartition(repartition), new_filter)
            }
            LogicalPlan::Distinct(distinct) => {
                let new_filter =
                    Filter::try_new(filter.predicate, Arc::clone(distinct.input()))
                        .map(LogicalPlan::Filter)?;
                insert_below(LogicalPlan::Distinct(distinct), new_filter)
            }
            LogicalPlan::Sort(sort) => {
                let new_filter =
                    Filter::try_new(filter.predicate, Arc::clone(&sort.input))
                        .map(LogicalPlan::Filter)?;
                insert_below(LogicalPlan::Sort(sort), new_filter)
            }
            LogicalPlan::SubqueryAlias(subquery_alias) => {
                let mut replace_map = HashMap::new();
                for (i, (qualifier, field)) in
                    subquery_alias.input.schema().iter().enumerate()
                {
                    let (sub_qualifier, sub_field) =
                        subquery_alias.schema.qualified_field(i);
                    replace_map.insert(
                        qualified_name(sub_qualifier, sub_field.name()),
                        Expr::Column(Column::new(qualifier.cloned(), field.name())),
                    );
                }
                let new_predicate = replace_cols_by_name(filter.predicate, &replace_map)?;

                let new_filter = LogicalPlan::Filter(Filter::try_new(
                    new_predicate,
                    Arc::clone(&subquery_alias.input),
                )?);
                insert_below(LogicalPlan::SubqueryAlias(subquery_alias), new_filter)
            }
            LogicalPlan::Projection(projection) => {
                let predicates = split_conjunction_owned(filter.predicate.clone());
                let (new_projection, keep_predicate) =
                    rewrite_projection(predicates, projection)?;
                if new_projection.transformed {
                    match keep_predicate {
                        None => Ok(new_projection),
                        Some(keep_predicate) => new_projection.map_data(|child_plan| {
                            Filter::try_new(keep_predicate, Arc::new(child_plan))
                                .map(LogicalPlan::Filter)
                        }),
                    }
                } else {
                    filter.input = Arc::new(new_projection.data);
                    Ok(Transformed::no(LogicalPlan::Filter(filter)))
                }
            }
            LogicalPlan::Unnest(mut unnest) => {
                let predicates = split_conjunction_owned(filter.predicate.clone());
                let mut non_unnest_predicates = vec![];
                let mut unnest_predicates = vec![];
                let mut unnest_struct_columns = vec![];

                for idx in &unnest.struct_type_columns {
                    let (sub_qualifier, field) =
                        unnest.input.schema().qualified_field(*idx);
                    let field_name = field.name().clone();

                    if let DataType::Struct(children) = field.data_type() {
                        for child in children {
                            let child_name = child.name().clone();
                            unnest_struct_columns.push(Column::new(
                                sub_qualifier.cloned(),
                                format!("{field_name}.{child_name}"),
                            ));
                        }
                    }
                }

                for predicate in predicates {
                    // collect all the Expr::Column in predicate recursively
                    let mut accum: HashSet<Column> = HashSet::new();
                    expr_to_columns(&predicate, &mut accum)?;

                    let contains_list_columns =
                        unnest.list_type_columns.iter().any(|(_, unnest_list)| {
                            accum.contains(&unnest_list.output_column)
                        });
                    let contains_struct_columns =
                        unnest_struct_columns.iter().any(|c| accum.contains(c));

                    if contains_list_columns || contains_struct_columns {
                        unnest_predicates.push(predicate);
                    } else {
                        non_unnest_predicates.push(predicate);
                    }
                }

                // Unnest predicates should not be pushed down.
                // If no non-unnest predicates exist, early return
                if non_unnest_predicates.is_empty() {
                    filter.input = Arc::new(LogicalPlan::Unnest(unnest));
                    return Ok(Transformed::no(LogicalPlan::Filter(filter)));
                }

                // Push down non-unnest filter predicate
                // Unnest
                //   Unnest Input (Projection)
                // -> rewritten to
                // Unnest
                //   Filter
                //     Unnest Input (Projection)

                let unnest_input = std::mem::take(&mut unnest.input);

                let filter_with_unnest_input = LogicalPlan::Filter(Filter::try_new(
                    conjunction(non_unnest_predicates).unwrap(), // Safe to unwrap since non_unnest_predicates is not empty.
                    unnest_input,
                )?);

                // Directly assign new filter plan as the new unnest's input.
                // The new filter plan will go through another rewrite pass since the rule itself
                // is applied recursively to all the child from top to down
                let unnest_plan =
                    insert_below(LogicalPlan::Unnest(unnest), filter_with_unnest_input)?;

                match conjunction(unnest_predicates) {
                    None => Ok(unnest_plan),
                    Some(predicate) => Ok(Transformed::yes(LogicalPlan::Filter(
                        Filter::try_new(predicate, Arc::new(unnest_plan.data))?,
                    ))),
                }
            }
            LogicalPlan::Union(ref union) => {
                let mut inputs = Vec::with_capacity(union.inputs.len());
                for input in &union.inputs {
                    let mut replace_map = HashMap::new();
                    for (i, (qualifier, field)) in input.schema().iter().enumerate() {
                        let (union_qualifier, union_field) =
                            union.schema.qualified_field(i);
                        replace_map.insert(
                            qualified_name(union_qualifier, union_field.name()),
                            Expr::Column(Column::new(qualifier.cloned(), field.name())),
                        );
                    }

                    let push_predicate =
                        replace_cols_by_name(filter.predicate.clone(), &replace_map)?;
                    inputs.push(Arc::new(LogicalPlan::Filter(Filter::try_new(
                        push_predicate,
                        Arc::clone(input),
                    )?)))
                }
                Ok(Transformed::yes(LogicalPlan::Union(Union {
                    inputs,
                    schema: Arc::clone(&plan_schema),
                })))
            }
            LogicalPlan::Aggregate(agg) => {
                // We can push down Predicate which in groupby_expr.
                let group_expr_columns = agg
                    .group_expr
                    .iter()
                    .map(|e| {
                        let (relation, name) = e.qualified_name();
                        Column::new(relation, name)
                    })
                    .collect::<HashSet<_>>();

                let predicates = split_conjunction_owned(filter.predicate);

                let mut keep_predicates = vec![];
                let mut push_predicates = vec![];
                for expr in predicates {
                    let cols = expr.column_refs();
                    if cols.iter().all(|c| group_expr_columns.contains(c)) {
                        push_predicates.push(expr);
                    } else {
                        keep_predicates.push(expr);
                    }
                }

                // As for plan Filter: Column(a+b) > 0 -- Agg: groupby:[Column(a)+Column(b)]
                // After push, we need to replace `a+b` with Column(a)+Column(b)
                // So we need create a replace_map, add {`a+b` --> Expr(Column(a)+Column(b))}
                let mut replace_map = HashMap::new();
                for expr in &agg.group_expr {
                    replace_map.insert(expr.schema_name().to_string(), expr.clone());
                }
                let replaced_push_predicates = push_predicates
                    .into_iter()
                    .map(|expr| replace_cols_by_name(expr, &replace_map))
                    .collect::<Result<Vec<_>>>()?;

                let agg_input = Arc::clone(&agg.input);
                Transformed::yes(LogicalPlan::Aggregate(agg))
                    .transform_data(|new_plan| {
                        // If we have a filter to push, we push it down to the input of the aggregate
                        if let Some(predicate) = conjunction(replaced_push_predicates) {
                            let new_filter = make_filter(predicate, agg_input)?;
                            insert_below(new_plan, new_filter)
                        } else {
                            Ok(Transformed::no(new_plan))
                        }
                    })?
                    .map_data(|child_plan| {
                        // if there are any remaining predicates we can't push, add them
                        // back as a filter
                        if let Some(predicate) = conjunction(keep_predicates) {
                            make_filter(predicate, Arc::new(child_plan))
                        } else {
                            Ok(child_plan)
                        }
                    })
            }
            // Tries to push filters based on the partition key(s) of the window function(s) used.
            // Example:
            //   Before:
            //     Filter: (a > 1) and (b > 1) and (c > 1)
            //      Window: func() PARTITION BY [a] ...
            //   ---
            //   After:
            //     Filter: (b > 1) and (c > 1)
            //      Window: func() PARTITION BY [a] ...
            //        Filter: (a > 1)
            LogicalPlan::Window(window) => {
                // Retrieve the set of potential partition keys where we can push filters by.
                // Unlike aggregations, where there is only one statement per SELECT, there can be
                // multiple window functions, each with potentially different partition keys.
                // Therefore, we need to ensure that any potential partition key returned is used in
                // ALL window functions. Otherwise, filters cannot be pushed by through that column.
                let extract_partition_keys = |func: &WindowFunction| {
                    func.params
                        .partition_by
                        .iter()
                        .map(|c| {
                            let (relation, name) = c.qualified_name();
                            Column::new(relation, name)
                        })
                        .collect::<HashSet<_>>()
                };
                let potential_partition_keys = window
                    .window_expr
                    .iter()
                    .map(|e| {
                        match e {
                            Expr::WindowFunction(window_func) => {
                                extract_partition_keys(window_func)
                            }
                            Expr::Alias(alias) => {
                                if let Expr::WindowFunction(window_func) =
                                    alias.expr.as_ref()
                                {
                                    extract_partition_keys(window_func)
                                } else {
                                    // window functions expressions are only Expr::WindowFunction
                                    unreachable!()
                                }
                            }
                            _ => {
                                // window functions expressions are only Expr::WindowFunction
                                unreachable!()
                            }
                        }
                    })
                    // performs the set intersection of the partition keys of all window functions,
                    // returning only the common ones
                    .reduce(|a, b| &a & &b)
                    .unwrap_or_default();

                let predicates = split_conjunction_owned(filter.predicate);
                let mut keep_predicates = vec![];
                let mut push_predicates = vec![];
                for expr in predicates {
                    let cols = expr.column_refs();
                    if cols.iter().all(|c| potential_partition_keys.contains(c)) {
                        push_predicates.push(expr);
                    } else {
                        keep_predicates.push(expr);
                    }
                }

                // Unlike with aggregations, there are no cases where we have to replace, e.g.,
                // `a+b` with Column(a)+Column(b). This is because partition expressions are not
                // available as standalone columns to the user. For example, while an aggregation on
                // `a+b` becomes Column(a + b), in a window partition it becomes
                // `func() PARTITION BY [a + b] ...`. Thus, filters on expressions always remain in
                // place, so we can use `push_predicates` directly. This is consistent with other
                // optimizers, such as the one used by Postgres.

                let window_input = Arc::clone(&window.input);
                Transformed::yes(LogicalPlan::Window(window))
                    .transform_data(|new_plan| {
                        // If we have a filter to push, we push it down to the input of the window
                        if let Some(predicate) = conjunction(push_predicates) {
                            let new_filter = make_filter(predicate, window_input)?;
                            insert_below(new_plan, new_filter)
                        } else {
                            Ok(Transformed::no(new_plan))
                        }
                    })?
                    .map_data(|child_plan| {
                        // if there are any remaining predicates we can't push, add them
                        // back as a filter
                        if let Some(predicate) = conjunction(keep_predicates) {
                            make_filter(predicate, Arc::new(child_plan))
                        } else {
                            Ok(child_plan)
                        }
                    })
            }
            LogicalPlan::Join(join) => push_down_join(join, Some(&filter.predicate)),
            LogicalPlan::TableScan(scan) => {
                let filter_predicates = split_conjunction(&filter.predicate);

                let (volatile_filters, non_volatile_filters): (Vec<&Expr>, Vec<&Expr>) =
                    filter_predicates
                        .into_iter()
                        .partition(|pred| pred.is_volatile());

                // Check which non-volatile filters are supported by source
                let supported_filters = scan
                    .source
                    .supports_filters_pushdown(non_volatile_filters.as_slice())?;
                assert_eq_or_internal_err!(
                    non_volatile_filters.len(),
                    supported_filters.len(),
                    "Vec returned length: {} from supports_filters_pushdown is not the same size as the filters passed, which length is: {}",
                    supported_filters.len(),
                    non_volatile_filters.len()
                );

                // Compose scan filters from non-volatile filters of `Exact` or `Inexact` pushdown type
                let zip = non_volatile_filters.into_iter().zip(supported_filters);

                let new_scan_filters = zip
                    .clone()
                    .filter(|(_, res)| res != &TableProviderFilterPushDown::Unsupported)
                    .map(|(pred, _)| pred);

                // Add new scan filters
                let new_scan_filters: Vec<Expr> = scan
                    .filters
                    .iter()
                    .chain(new_scan_filters)
                    .unique()
                    .cloned()
                    .collect();

                // Compose predicates to be of `Unsupported` or `Inexact` pushdown type, and also include volatile filters
                let new_predicate: Vec<Expr> = zip
                    .filter(|(_, res)| res != &TableProviderFilterPushDown::Exact)
                    .map(|(pred, _)| pred)
                    .chain(volatile_filters)
                    .cloned()
                    .collect();

                let new_scan = LogicalPlan::TableScan(TableScan {
                    filters: new_scan_filters,
                    ..scan
                });

                Transformed::yes(new_scan).transform_data(|new_scan| {
                    if let Some(predicate) = conjunction(new_predicate) {
                        make_filter(predicate, Arc::new(new_scan)).map(Transformed::yes)
                    } else {
                        Ok(Transformed::no(new_scan))
                    }
                })
            }
            LogicalPlan::Extension(extension_plan) => {
                // This check prevents the Filter from being removed when the extension node has no children,
                // so we return the original Filter unchanged.
                if extension_plan.node.inputs().is_empty() {
                    filter.input = Arc::new(LogicalPlan::Extension(extension_plan));
                    return Ok(Transformed::no(LogicalPlan::Filter(filter)));
                }
                let prevent_cols =
                    extension_plan.node.prevent_predicate_push_down_columns();

                // determine if we can push any predicates down past the extension node

                // each element is true for push, false to keep
                let predicate_push_or_keep = split_conjunction(&filter.predicate)
                    .iter()
                    .map(|expr| {
                        let cols = expr.column_refs();
                        if !extension_plan.node.supports_predicate_pushdown(expr)
                            || cols.iter().any(|c| prevent_cols.contains(&c.name)) {
                            Ok(false) // No push (keep)
                        } else {
                            Ok(true) // push
                        }
                    })
                    .collect::<Result<Vec<_>>>()?;

                // all predicates are kept, no changes needed
                if predicate_push_or_keep.iter().all(|&x| !x) {
                    filter.input = Arc::new(LogicalPlan::Extension(extension_plan));
                    return Ok(Transformed::no(LogicalPlan::Filter(filter)));
                }

                // going to push some predicates down, so split the predicates
                let mut keep_predicates = vec![];
                let mut push_predicates = vec![];
                for (push, expr) in predicate_push_or_keep
                    .into_iter()
                    .zip(split_conjunction_owned(filter.predicate).into_iter())
                {
                    if !push {
                        keep_predicates.push(expr);
                    } else {
                        push_predicates.push(expr);
                    }
                }

                let new_children = match conjunction(push_predicates) {
                    Some(predicate) => extension_plan
                        .node
                        .inputs()
                        .into_iter()
                        .map(|child| {
                            Ok(LogicalPlan::Filter(Filter::try_new(
                                predicate.clone(),
                                Arc::new(child.clone()),
                            )?))
                        })
                        .collect::<Result<Vec<_>>>()?,
                    None => extension_plan.node.inputs().into_iter().cloned().collect(),
                };
                // extension with new inputs.
                let child_plan = LogicalPlan::Extension(extension_plan);
                let new_extension =
                    child_plan.with_new_exprs(child_plan.expressions(), new_children)?;

                let new_plan = match conjunction(keep_predicates) {
                    Some(predicate) => LogicalPlan::Filter(Filter::try_new(
                        predicate,
                        Arc::new(new_extension),
                    )?),
                    None => new_extension,
                };
                Ok(Transformed::yes(new_plan))
            }
            child => {
                filter.input = Arc::new(child);
                Ok(Transformed::no(LogicalPlan::Filter(filter)))
            }
        }
    }
}

/// Attempts to push `predicate` into a `FilterExec` below `projection
///
/// # Returns
/// (plan, remaining_predicate)
///
/// `plan` is a LogicalPlan for `projection` with possibly a new FilterExec below it.
/// `remaining_predicate` is any part of the predicate that could not be pushed down
///
/// # Args
/// - predicates: Split predicates like `[foo=5, bar=6]`
/// - projection: The target projection plan to push down the predicates
///
/// # Example
///
/// Pushing a predicate like `foo=5 AND bar=6` with an input plan like this:
///
/// ```text
/// Projection(foo, c+d as bar)
/// ```
///
/// Might result in returning `remaining_predicate` of `bar=6` and a plan like
///
/// ```text
/// Projection(foo, c+d as bar)
///  Filter(foo=5)
///   ...
/// ```
fn rewrite_projection(
    predicates: Vec<Expr>,
    mut projection: Projection,
) -> Result<(Transformed<LogicalPlan>, Option<Expr>)> {
    // Partition projection expressions into non-pushable vs pushable.
    // Non-pushable expressions are volatile (must not be duplicated) or
    // MoveTowardsLeafNodes (cheap expressions like get_field where re-inlining
    // into a filter causes optimizer instability — ExtractLeafExpressions will
    // undo the push-down, creating an infinite loop that runs until the
    // iteration limit is hit).
    let (non_pushable_map, pushable_map): (HashMap<_, _>, HashMap<_, _>) = projection
        .schema
        .iter()
        .zip(projection.expr.iter())
        .map(|((qualifier, field), expr)| {
            // strip alias, as they should not be part of filters
            let expr = expr.clone().unalias();

            (qualified_name(qualifier, field.name()), expr)
        })
        .partition(|(_, value)| {
            value.is_volatile()
                || value.placement() == ExpressionPlacement::MoveTowardsLeafNodes
        });

    let mut push_predicates = vec![];
    let mut keep_predicates = vec![];
    for expr in predicates {
        if contain(&expr, &non_pushable_map) {
            keep_predicates.push(expr);
        } else {
            push_predicates.push(expr);
        }
    }

    match conjunction(push_predicates) {
        Some(expr) => {
            // re-write all filters based on this projection
            // E.g. in `Filter: b\n  Projection: a > 1 as b`, we can swap them, but the filter must be "a > 1"
            let new_filter = LogicalPlan::Filter(Filter::try_new(
                replace_cols_by_name(expr, &pushable_map)?,
                std::mem::take(&mut projection.input),
            )?);

            projection.input = Arc::new(new_filter);

            Ok((
                Transformed::yes(LogicalPlan::Projection(projection)),
                conjunction(keep_predicates),
            ))
        }
        None => Ok((
            Transformed::no(LogicalPlan::Projection(projection)),
            conjunction(keep_predicates),
        )),
    }
}

/// Creates a new LogicalPlan::Filter node.
pub fn make_filter(predicate: Expr, input: Arc<LogicalPlan>) -> Result<LogicalPlan> {
    Filter::try_new(predicate, input).map(LogicalPlan::Filter)
}

/// Replace the existing child of the single input node with `new_child`.
///
/// Starting:
/// ```text
/// plan
///   child
/// ```
///
/// Ending:
/// ```text
/// plan
///   new_child
/// ```
fn insert_below(
    plan: LogicalPlan,
    new_child: LogicalPlan,
) -> Result<Transformed<LogicalPlan>> {
    let mut new_child = Some(new_child);
    let transformed_plan = plan.map_children(|_child| {
        if let Some(new_child) = new_child.take() {
            Ok(Transformed::yes(new_child))
        } else {
            // already took the new child
            internal_err!("node had more than one input")
        }
    })?;

    // make sure we did the actual replacement
    assert_or_internal_err!(new_child.is_none(), "node had no inputs");

    Ok(transformed_plan)
}

impl PushDownFilter {
    #[expect(missing_docs)]
    pub fn new() -> Self {
        Self {}
    }
}

/// replaces columns by its name on the projection.
pub fn replace_cols_by_name(
    e: Expr,
    replace_map: &HashMap<String, Expr>,
) -> Result<Expr> {
    e.transform_up(|expr| {
        Ok(if let Expr::Column(c) = &expr {
            match replace_map.get(&c.flat_name()) {
                Some(new_c) => Transformed::yes(new_c.clone()),
                None => Transformed::no(expr),
            }
        } else {
            Transformed::no(expr)
        })
    })
    .data()
}

/// check whether the expression uses the columns in `check_map`.
fn contain(e: &Expr, check_map: &HashMap<String, Expr>) -> bool {
    let mut is_contain = false;
    e.apply(|expr| {
        Ok(if let Expr::Column(c) = &expr {
            match check_map.get(&c.flat_name()) {
                Some(_) => {
                    is_contain = true;
                    TreeNodeRecursion::Stop
                }
                None => TreeNodeRecursion::Continue,
            }
        } else {
            TreeNodeRecursion::Continue
        })
    })
    .unwrap();
    is_contain
}
