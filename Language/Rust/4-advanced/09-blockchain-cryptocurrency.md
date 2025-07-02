# Blockchain and Cryptocurrency Development

## Overview
Rust is increasingly popular for blockchain development due to its performance, safety, and low-level control. This guide covers blockchain fundamentals and cryptocurrency project development.

## Core Blockchain Concepts

### Basic Blockchain Structure
```rust
use sha2::{Digest, Sha256};
use serde::{Deserialize, Serialize};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Block {
    pub index: u64,
    pub timestamp: u64,
    pub data: String,
    pub previous_hash: String,
    pub hash: String,
    pub nonce: u64,
}

impl Block {
    pub fn new(index: u64, data: String, previous_hash: String) -> Self {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        let mut block = Block {
            index,
            timestamp,
            data,
            previous_hash,
            hash: String::new(),
            nonce: 0,
        };
        
        block.hash = block.calculate_hash();
        block
    }
    
    pub fn calculate_hash(&self) -> String {
        let input = format!("{}{}{}{}{}", 
            self.index, self.timestamp, self.data, self.previous_hash, self.nonce);
        format!("{:x}", Sha256::digest(input.as_bytes()))
    }
    
    pub fn mine_block(&mut self, difficulty: usize) {
        let target = "0".repeat(difficulty);
        while &self.hash[..difficulty] != target {
            self.nonce += 1;
            self.hash = self.calculate_hash();
        }
        println!("Block mined: {}", self.hash);
    }
}
```

### Simple Blockchain Implementation
```rust
pub struct Blockchain {
    pub chain: Vec<Block>,
    pub difficulty: usize,
}

impl Blockchain {
    pub fn new() -> Self {
        let mut blockchain = Blockchain {
            chain: Vec::new(),
            difficulty: 2,
        };
        blockchain.create_genesis_block();
        blockchain
    }
    
    fn create_genesis_block(&mut self) {
        let genesis = Block::new(0, "Genesis Block".to_string(), "0".to_string());
        self.chain.push(genesis);
    }
    
    pub fn add_block(&mut self, data: String) {
        let previous_block = self.chain.last().unwrap();
        let mut new_block = Block::new(
            previous_block.index + 1,
            data,
            previous_block.hash.clone(),
        );
        new_block.mine_block(self.difficulty);
        self.chain.push(new_block);
    }
    
    pub fn is_chain_valid(&self) -> bool {
        for i in 1..self.chain.len() {
            let current = &self.chain[i];
            let previous = &self.chain[i - 1];
            
            if current.hash != current.calculate_hash() {
                return false;
            }
            
            if current.previous_hash != previous.hash {
                return false;
            }
        }
        true
    }
}
```

## Cryptocurrency Features

### Transaction System
```rust
use ed25519_dalek::{Keypair, PublicKey, Signature, Signer, Verifier};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Transaction {
    pub from: String,
    pub to: String,
    pub amount: f64,
    pub signature: Option<String>,
}

impl Transaction {
    pub fn new(from: String, to: String, amount: f64) -> Self {
        Transaction {
            from,
            to,
            amount,
            signature: None,
        }
    }
    
    pub fn calculate_hash(&self) -> String {
        let input = format!("{}{}{}", self.from, self.to, self.amount);
        format!("{:x}", Sha256::digest(input.as_bytes()))
    }
    
    pub fn sign_transaction(&mut self, signing_key: &Keypair) {
        let hash = self.calculate_hash();
        let signature = signing_key.sign(hash.as_bytes());
        self.signature = Some(hex::encode(signature.to_bytes()));
    }
    
    pub fn is_valid(&self, public_key: &PublicKey) -> bool {
        if let Some(sig_str) = &self.signature {
            if let Ok(sig_bytes) = hex::decode(sig_str) {
                if let Ok(signature) = Signature::from_bytes(&sig_bytes) {
                    let hash = self.calculate_hash();
                    return public_key.verify(hash.as_bytes(), &signature).is_ok();
                }
            }
        }
        false
    }
}
```

### Wallet Implementation
```rust
use std::collections::HashMap;

pub struct Wallet {
    pub keypair: Keypair,
    pub public_key: PublicKey,
}

impl Wallet {
    pub fn new() -> Self {
        let keypair = Keypair::generate(&mut rand::thread_rng());
        let public_key = keypair.public;
        
        Wallet { keypair, public_key }
    }
    
    pub fn get_balance(&self, blockchain: &Blockchain) -> f64 {
        let mut balance = 0.0;
        let address = hex::encode(self.public_key.as_bytes());
        
        for block in &blockchain.chain {
            // In a real implementation, blocks would contain transactions
            // This is simplified for demonstration
        }
        
        balance
    }
    
    pub fn send_money(&self, to: String, amount: f64) -> Transaction {
        let from = hex::encode(self.public_key.as_bytes());
        let mut transaction = Transaction::new(from, to, amount);
        transaction.sign_transaction(&self.keypair);
        transaction
    }
}
```

## Popular Rust Blockchain Frameworks

### Substrate Framework
```rust
// Example Substrate pallet structure
use frame_support::{
    decl_module, decl_storage, decl_event, decl_error,
    traits::{Get, Randomness},
    dispatch::DispatchResult,
};
use frame_system::ensure_signed;
use sp_std::vec::Vec;

pub trait Trait: frame_system::Trait {
    type Event: From<Event<Self>> + Into<<Self as frame_system::Trait>::Event>;
}

decl_storage! {
    trait Store for Module<T: Trait> as TemplateModule {
        Something get(fn something): Option<u32>;
        Values get(fn values): map hasher(blake2_128_concat) T::AccountId => u32;
    }
}

decl_event!(
    pub enum Event<T> where AccountId = <T as frame_system::Trait>::AccountId {
        SomethingStored(u32, AccountId),
    }
);

decl_error! {
    pub enum Error for Module<T: Trait> {
        NoneValue,
        StorageOverflow,
    }
}

decl_module! {
    pub struct Module<T: Trait> for enum Call where origin: T::Origin {
        type Error = Error<T>;
        fn deposit_event() = default;
        
        #[weight = 10_000]
        pub fn do_something(origin, something: u32) -> DispatchResult {
            let who = ensure_signed(origin)?;
            Something::put(something);
            Self::deposit_event(RawEvent::SomethingStored(something, who));
            Ok(())
        }
    }
}
```

### Smart Contract with ink!
```rust
#![cfg_attr(not(feature = "std"), no_std)]

use ink_lang as ink;

#[ink::contract]
mod simple_token {
    use ink_storage::{
        traits::SpreadAllocate,
        Mapping,
    };

    #[ink(storage)]
    #[derive(SpreadAllocate)]
    pub struct SimpleToken {
        total_supply: Balance,
        balances: Mapping<AccountId, Balance>,
        allowances: Mapping<(AccountId, AccountId), Balance>,
    }

    #[ink(event)]
    pub struct Transfer {
        #[ink(topic)]
        from: Option<AccountId>,
        #[ink(topic)]
        to: Option<AccountId>,
        value: Balance,
    }

    impl SimpleToken {
        #[ink(constructor)]
        pub fn new(initial_supply: Balance) -> Self {
            ink_lang::utils::initialize_contract(|contract: &mut Self| {
                let caller = Self::env().caller();
                contract.balances.insert(&caller, &initial_supply);
                contract.total_supply = initial_supply;
            })
        }

        #[ink(message)]
        pub fn total_supply(&self) -> Balance {
            self.total_supply
        }

        #[ink(message)]
        pub fn balance_of(&self, owner: AccountId) -> Balance {
            self.balance_of_impl(&owner)
        }

        #[ink(message)]
        pub fn transfer(&mut self, to: AccountId, value: Balance) -> bool {
            let from = self.env().caller();
            self.transfer_from_to(&from, &to, value)
        }

        fn transfer_from_to(&mut self, from: &AccountId, to: &AccountId, value: Balance) -> bool {
            let from_balance = self.balance_of_impl(from);
            if from_balance < value {
                return false;
            }

            self.balances.insert(from, &(from_balance - value));
            let to_balance = self.balance_of_impl(to);
            self.balances.insert(to, &(to_balance + value));

            self.env().emit_event(Transfer {
                from: Some(*from),
                to: Some(*to),
                value,
            });

            true
        }

        fn balance_of_impl(&self, owner: &AccountId) -> Balance {
            self.balances.get(owner).unwrap_or_default()
        }
    }
}
```

## Common Libraries and Tools

### Essential Crates
```toml
[dependencies]
# Cryptography
sha2 = "0.10"
ed25519-dalek = "1.0"
ring = "0.16"
hex = "0.4"

# Serialization
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
bincode = "1.3"

# Networking
tokio = { version = "1.0", features = ["full"] }
libp2p = "0.45"

# Blockchain frameworks
substrate = "3.0"
ink_lang = "3.4"
```

### P2P Networking Example
```rust
use libp2p::{
    noise, tcp, yamux, PeerId, Transport,
    identity, swarm::{Swarm, SwarmBuilder},
};

async fn create_peer() -> Result<Swarm<()>, Box<dyn std::error::Error>> {
    let local_key = identity::Keypair::generate_ed25519();
    let local_peer_id = PeerId::from(local_key.public());
    
    let transport = tcp::TcpConfig::new()
        .upgrade(upgrade::Version::V1)
        .authenticate(noise::NoiseConfig::xx(local_key).into_authenticated())
        .multiplex(yamux::YamuxConfig::default())
        .boxed();

    let behaviour = (); // Your custom behaviour here
    
    let swarm = SwarmBuilder::new(transport, behaviour, local_peer_id)
        .build();
    
    Ok(swarm)
}
```

## Best Practices

1. **Security First**: Always validate inputs and use proven cryptographic libraries
2. **Testing**: Extensive unit and integration testing for financial applications
3. **Performance**: Optimize for low latency and high throughput
4. **Auditability**: Write clear, well-documented code for security audits
5. **Gas Optimization**: For smart contracts, minimize computational complexity

## Project Ideas

1. **Simple Cryptocurrency**: Build a complete blockchain with mining and transactions
2. **DeFi Protocol**: Create lending/borrowing smart contracts
3. **NFT Marketplace**: Build an NFT trading platform
4. **Decentralized Exchange**: Implement an AMM-based DEX
5. **Cross-chain Bridge**: Build asset transfer between blockchains

## Learning Resources

- Substrate Developer Hub
- ink! Documentation
- Polkadot Wiki
- Ethereum Development with Rust
- Blockchain programming tutorials
