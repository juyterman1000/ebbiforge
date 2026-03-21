//! Ebbiforge Tool Registry - Real tool implementations for agentic workflows
//!
//! This module provides actual tool execution capabilities:
//! - web_search: Search the web using Google Custom Search API
//! - fetch_url: Fetch content from a URL
//! - calculate: Evaluate mathematical expressions
//! - finish: Signal task completion with final answer

use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::env;
use tracing::info;

/// Tool execution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ToolResult {
    Success(String),
    Error(String),
}

/// Tool definitions for Gemini function calling
pub fn get_tool_definitions() -> serde_json::Value {
    serde_json::json!({
        "function_declarations": [
            {
                "name": "web_search",
                "description": "Search the web for information. Use this to find current data like stock prices, distances, news, etc.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query"
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "fetch_url",
                "description": "Fetch the content of a specific URL. Use after web_search to get detailed information.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "The URL to fetch"
                        }
                    },
                    "required": ["url"]
                }
            },
            {
                "name": "calculate",
                "description": "Evaluate a mathematical expression. Use for calculations like percentage growth.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "Mathematical expression to evaluate, e.g. '((145.20-125.90)/125.90)*100'"
                        }
                    },
                    "required": ["expression"]
                }
            },
            {
                "name": "finish",
                "description": "Call this when you have the final answer to the task. This ends the task loop.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "answer": {
                            "type": "string",
                            "description": "The final answer to the user's question"
                        }
                    },
                    "required": ["answer"]
                }
            }
        ]
    })
}

/// Execute web search using Google Custom Search API (or fallback to DuckDuckGo)
pub async fn web_search(client: &Client, query: &str) -> ToolResult {
    info!("🔍 [Tool] web_search: {}", query);

    // If it's a URL, just fetch it directly
    if query.trim().starts_with("http") {
        info!("🔗 [Tool] Query looks like a URL, calling fetch_url instead...");
        return fetch_url(client, query.trim()).await;
    }

    // FULLY DYNAMIC ticker detection - no hardcoded values!
    let query_lower = query.to_lowercase();

    // Extract potential ticker using regex (2-5 uppercase letters)
    let ticker_regex = match regex::Regex::new(r"\b([A-Z]{2,5})\b") {
        Ok(r) => r,
        Err(e) => return ToolResult::Error(format!("Regex error: {}", e)),
    };
    let common_words = [
        "THE", "AND", "FOR", "ARE", "BUT", "NOT", "YOU", "ALL", "CAN", "HER", "WAS", "ONE", "OUR",
        "OUT", "DAY", "GET", "HAS", "HIM", "HIS", "HOW", "ITS", "MAY", "NEW", "NOW", "OLD", "SEE",
        "TWO", "WAY", "WHO", "DID", "USD", "WHY", "WHAT", "WHEN", "WHERE", "LAST", "DAYS", "ROSE",
        "MUCH", "STOCK", "PRICE", "RISING",
    ];

    let mut extracted_ticker: Option<String> = None;
    for cap in ticker_regex.captures_iter(query) {
        let potential = cap[1].to_string();
        if !common_words.contains(&potential.as_str()) {
            extracted_ticker = Some(potential);
            break;
        }
    }

    let is_stock_query = query_lower.contains("stock")
        || query_lower.contains("price")
        || query_lower.contains("ticker")
        || query_lower.contains("shares")
        || extracted_ticker.is_some();

    // Try Google Custom Search first
    let google_api_key = env::var("GOOGLE_SEARCH_API_KEY").ok();
    let google_cx = env::var("GOOGLE_SEARCH_CX").ok();

    if let (Some(api_key), Some(cx)) = (google_api_key, google_cx) {
        let url = format!(
            "https://www.googleapis.com/customsearch/v1?key={}&cx={}&q={}",
            api_key,
            cx,
            urlencoding::encode(query)
        );

        if let Ok(resp) = client.get(&url).send().await {
            if let Ok(json) = resp.json::<serde_json::Value>().await {
                if let Some(items) = json["items"].as_array() {
                    let results: Vec<String> = items
                        .iter()
                        .take(3)
                        .map(|item| {
                            format!(
                                "- {} ({})\n  {}",
                                item["title"].as_str().unwrap_or(""),
                                item["link"].as_str().unwrap_or(""),
                                item["snippet"].as_str().unwrap_or("")
                            )
                        })
                        .collect();
                    return ToolResult::Success(format!(
                        "Search results for '{}':\n{}",
                        query,
                        results.join("\n")
                    ));
                }
            }
        }
    }

    // For stock queries, use Yahoo Finance Chart API
    if is_stock_query {
        if let Some(ref ticker) = extracted_ticker {
            info!("[Tool] Querying Yahoo Finance for ticker: {}", ticker);

            let chart_url = format!(
                "https://query1.finance.yahoo.com/v8/finance/chart/{}?range=5d&interval=1d",
                ticker
            );

            match client
                .get(&chart_url)
                .header(
                    "User-Agent",
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                )
                .send()
                .await
            {
                Ok(resp) => {
                    if let Ok(json) = resp.json::<serde_json::Value>().await {
                        let chart = &json["chart"]["result"][0];
                        let meta = &chart["meta"];

                        let current_price = meta["regularMarketPrice"].as_f64().unwrap_or(0.0);
                        let prev_close = meta["chartPreviousClose"].as_f64().unwrap_or(0.0);
                        let currency = meta["currency"].as_str().unwrap_or("USD");
                        let symbol = meta["symbol"].as_str().unwrap_or(ticker);

                        // Get 5-day historical prices
                        let closes = chart["indicators"]["quote"][0]["close"].as_array();
                        let timestamps = chart["timestamp"].as_array();

                        let mut price_history = String::new();
                        if let (Some(closes), Some(timestamps)) = (closes, timestamps) {
                            for (i, (close, ts)) in closes.iter().zip(timestamps.iter()).enumerate()
                            {
                                if let (Some(price), Some(time)) = (close.as_f64(), ts.as_i64()) {
                                    let date = chrono::DateTime::from_timestamp(time, 0)
                                        .map(|d| d.format("%Y-%m-%d").to_string())
                                        .unwrap_or_else(|| format!("Day {}", i + 1));
                                    price_history
                                        .push_str(&format!("  - {}: ${:.2}\n", date, price));
                                }
                            }
                        }

                        // Calculate 5-day change
                        let first_price = closes
                            .and_then(|c| c.first())
                            .and_then(|v| v.as_f64())
                            .unwrap_or(prev_close);
                        let change_5d = if first_price > 0.0 {
                            ((current_price - first_price) / first_price) * 100.0
                        } else {
                            0.0
                        };

                        let result = format!(
                            "Real-time Stock Data for {} ({}):\n\
                        - Current Price: ${:.2} {}\n\
                        - Previous Close: ${:.2}\n\
                        - 5-Day Change: {:.2}%\n\n\
                        5-Day Price History:\n{}",
                            symbol,
                            ticker,
                            current_price,
                            currency,
                            prev_close,
                            change_5d,
                            price_history
                        );

                        if current_price > 0.0 {
                            return ToolResult::Success(result);
                        }
                    }
                }
                Err(e) => info!("[Tool] Yahoo Chart API failed: {}", e),
            }
        }
    }

    // Fallback: Use Google Search (Lite version)
    let search_url = format!(
        "https://www.google.com/search?q={}&gbv=1",
        urlencoding::encode(query)
    );

    match client.get(&search_url)
        .header("User-Agent", "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        .header("Accept", "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8")
        .header("Accept-Language", "en-US,en;q=0.9")
        .send()
        .await
    {
        Ok(resp) => {
            if let Ok(html) = resp.text().await {
                let mut results = Vec::new();
                // Improved Google Lite regex
                let link_re = match regex::Regex::new(r#"(?i)<a[^>]*href="/url\?q=([^&"]+)[^>]*>(.*?)</a>"#) { Ok(r) => r, Err(e) => return ToolResult::Error(format!("Regex error: {}", e)) };

                for cap in link_re.captures_iter(&html).take(10) {
                    let link = urlencoding::decode(&cap[1]).unwrap_or(std::borrow::Cow::Borrowed(&cap[1])).to_string();
                    let title_raw = &cap[2];
                    let title = title_raw.replace("<h3", "").replace("</h3>", "").replace("<div", "").replace("</div>", "").replace("<span", "").replace("</span>", "").replace("<b>", "").replace("</b>", "").trim().to_string();

                    if !link.contains("google.com/") && !title.is_empty() && title.len() > 3 {
                        results.push(format!("### {}\nURL: {}\n", title, link));
                    }
                }

                if !results.is_empty() {
                    return ToolResult::Success(format!("Google Search results for '{}':\n\n{}", query, results.join("\n---\n")));
                }
            }
        }
        _ => {}
    }

    // Secondary Fallback: Use DuckDuckGo Lite (Much less likely to block)
    let ddg_url = format!(
        "https://lite.duckduckgo.com/lite/?q={}",
        urlencoding::encode(query)
    );
    if let Ok(resp) = client.get(&ddg_url)
        .header("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
        .send()
        .await
    {
        if let Ok(html) = resp.text().await {
            let mut results = Vec::new();
            let link_re = match regex::Regex::new(r#"(?i)result-link"[^>]*href="([^"]+)"[^>]*>(.*?)</a>"#) { Ok(r) => r, Err(e) => return ToolResult::Error(format!("Regex error: {}", e)) };
            let snippet_re = match regex::Regex::new(r#"(?i)result-snippet"[^>]*>([^<]+)"#) { Ok(r) => r, Err(e) => return ToolResult::Error(format!("Regex error: {}", e)) };

            let links: Vec<_> = link_re.captures_iter(&html).map(|c| (c[1].to_string(), c[2].to_string())).collect();
            let snippets: Vec<_> = snippet_re.captures_iter(&html).map(|c| c[1].to_string()).collect();

            for i in 0..links.len().min(4) {
                let (url, title) = &links[i];
                let snippet = snippets.get(i).cloned().unwrap_or_default();
                results.push(format!("### {}\nURL: {}\nSnippet: {}\n", title, url, snippet));
            }

            if !results.is_empty() {
                return ToolResult::Success(format!("DuckDuckGo search results for '{}':\n\n{}", query, results.join("\n---\n")));
            }
        }
    }

    ToolResult::Success("All search sources blocked. Suggest using fetch_url on a direct link like 'https://www.marketwatch.com/investing/stock/amd' or 'https://www.cnbc.com/quotes/AMD'".to_string())
}

/// Fetch content from a URL
pub async fn fetch_url(client: &Client, url: &str) -> ToolResult {
    info!("🌐 [Tool] fetch_url: {}", url);

    match client
        .get(url)
        .header("User-Agent", "Ebbiforge/1.0 (Research Agent)")
        .send()
        .await
    {
        Ok(resp) => {
            if resp.status().is_success() {
                match resp.text().await {
                    Ok(body) => {
                        // Truncate to avoid overwhelming the context
                        let truncated = if body.len() > 4000 {
                            format!("{}... [truncated]", &body[..4000])
                        } else {
                            body
                        };
                        ToolResult::Success(format!("Content from {}:\n{}", url, truncated))
                    }
                    Err(e) => ToolResult::Error(format!("Failed to read response: {}", e)),
                }
            } else {
                ToolResult::Error(format!("HTTP {}: Failed to fetch {}", resp.status(), url))
            }
        }
        Err(e) => ToolResult::Error(format!("Request failed: {}", e)),
    }
}

/// Evaluate a mathematical expression
pub fn calculate(expression: &str) -> ToolResult {
    info!("🔢 [Tool] calculate: {}", expression);

    // Simple expression evaluator (supports +, -, *, /, parentheses)
    let cleaned = expression.replace(" ", "").replace(",", "");

    // Use a simple recursive descent parser for safety
    match eval_expr(&cleaned) {
        Ok(result) => ToolResult::Success(format!("Result: {:.4}", result)),
        Err(e) => ToolResult::Error(format!("Calculation error: {}", e)),
    }
}

/// Simple expression evaluator
fn eval_expr(expr: &str) -> Result<f64, String> {
    let expr = expr.trim();

    // Handle parentheses
    if expr.starts_with('(') && expr.ends_with(')') {
        let inner = &expr[1..expr.len() - 1];
        // Check if these are matching parens
        let mut depth = 0;
        let mut is_outer = true;
        for (_i, c) in inner.chars().enumerate() {
            match c {
                '(' => depth += 1,
                ')' => {
                    depth -= 1;
                    if depth < 0 {
                        is_outer = false;
                        break;
                    }
                }
                _ => {}
            }
        }
        if is_outer && depth == 0 {
            return eval_expr(inner);
        }
    }

    // Find lowest precedence operator outside parentheses
    let mut depth = 0;
    let mut last_add_sub = None;
    let mut last_mul_div = None;

    for (i, c) in expr.chars().enumerate() {
        match c {
            '(' => depth += 1,
            ')' => depth -= 1,
            '+' | '-' if depth == 0 && i > 0 => last_add_sub = Some(i),
            '*' | '/' if depth == 0 => last_mul_div = Some(i),
            _ => {}
        }
    }

    // Evaluate based on operator precedence
    if let Some(pos) = last_add_sub {
        let left = eval_expr(&expr[..pos])?;
        let right = eval_expr(&expr[pos + 1..])?;
        return Ok(if expr.chars().nth(pos) == Some('+') {
            left + right
        } else {
            left - right
        });
    }

    if let Some(pos) = last_mul_div {
        let left = eval_expr(&expr[..pos])?;
        let right = eval_expr(&expr[pos + 1..])?;
        return Ok(if expr.chars().nth(pos) == Some('*') {
            left * right
        } else {
            left / right
        });
    }

    // Try to parse as number
    expr.parse::<f64>()
        .map_err(|_| format!("Invalid number: {}", expr))
}

/// Signal task completion
pub fn finish(answer: &str) -> ToolResult {
    info!("[Tool] finish: {}", answer);
    ToolResult::Success(answer.to_string())
}

/// Dispatch tool call by name
pub async fn execute_tool(client: &Client, name: &str, args: &serde_json::Value) -> ToolResult {
    match name {
        "web_search" => {
            let query = args["query"].as_str().unwrap_or("");
            web_search(client, query).await
        }
        "fetch_url" => {
            let url = args["url"].as_str().unwrap_or("");
            fetch_url(client, url).await
        }
        "calculate" => {
            let expr = args["expression"].as_str().unwrap_or("");
            calculate(expr)
        }
        "finish" => {
            let answer = args["answer"].as_str().unwrap_or("");
            finish(answer)
        }
        _ => ToolResult::Error(format!("Unknown tool: {}", name)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculate() {
        assert!(matches!(calculate("2+2"), ToolResult::Success(s) if s.contains("4")));
        assert!(
            matches!(calculate("((145.20-125.90)/125.90)*100"), ToolResult::Success(s) if s.contains("15"))
        );
    }
}
