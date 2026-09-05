// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <functional>
#include <iostream>  // TODO: Remove
#include <mutex>
#include <string>
#include <vector>

#include "iree/compiler/embedding_api.h"
#include "iree_pjrt/common/compiler.h"

namespace iree::pjrt {

//===----------------------------------------------------------------------===//
// IREE compiler.
//===----------------------------------------------------------------------===//

namespace {

class MMapCompilerOutput : public CompilerOutput {
 public:
  MMapCompilerOutput(iree_compiler_output_t* output, void* data, size_t length)
      : output_(output), data_(data), length_(length) {}
  ~MMapCompilerOutput() { ireeCompilerOutputDestroy(output_); }
  void* GetData() { return data_; }
  size_t GetDataSize() { return length_; }

 private:
  iree_compiler_output_t* output_;
  void* data_;
  size_t length_;
};

using SessionRecycler = std::function<void(iree_compiler_session_t*)>;
class IREECompilerJob : public CompilerJob {
 public:
  // Takes ownership of both |session| and |inv|. On destruction, destroys
  // |inv| and passes |session| to the recycler (this can be used to implement
  // session pooling).
  IREECompilerJob(iree_compiler_session_t* session,
                  iree_compiler_invocation_t* inv,
                  SessionRecycler session_recycler)
      : session_(session), inv_(inv), session_recycler_(session_recycler) {}

  // Diagnostic sink for ireeCompilerInvocationEnableCallbackDiagnostics.
  //
  // Without this, a pipeline failure reaches the caller as an empty string:
  // GetErrorMessage() below has nothing to report unless a *flag* was
  // rejected (which is what sets error_), while everything the pass pipeline
  // says -- "failed to legalize operation 'stablehlo.cholesky'", and so on --
  // goes to the diagnostic engine instead. A PJRT client embedded in another
  // process then reports a compile failure with no reason attached, and the
  // actual text is somewhere in a stderr stream it may not even be showing.
  //
  // May be called from any thread, hence the lock.
  static void OnDiagnostic(enum iree_compiler_diagnostic_severity_t severity,
                           const char* message, size_t message_size,
                           void* user_data) {
    auto* self = static_cast<IREECompilerJob*>(user_data);
    const char* prefix;
    switch (severity) {
      case IREE_COMPILER_DIAGNOSTIC_SEVERITY_ERROR:
        prefix = "error: ";
        break;
      case IREE_COMPILER_DIAGNOSTIC_SEVERITY_WARNING:
        prefix = "warning: ";
        break;
      default:
        // Notes and remarks carry the "see also" context of an error, so they
        // are kept, just unlabelled.
        prefix = "";
        break;
    }
    std::lock_guard<std::mutex> guard(self->diagnostics_mutex_);
    self->diagnostics_.append(prefix);
    self->diagnostics_.append(message, message_size);
    self->diagnostics_.push_back('\n');
  }
  ~IREECompilerJob() {
    if (error_) {
      ireeCompilerErrorDestroy(error_);
    }
    for (auto* source : retained_sources_) {
      ireeCompilerSourceDestroy(source);
    }
    ireeCompilerInvocationDestroy(inv_);
    session_recycler_(session_);

    if (output_) {
      ireeCompilerOutputDestroy(output_);
    }
  }

  std::string GetErrorMessage() override {
    std::string message;
    if (error_) {
      message = ireeCompilerErrorGetMessage(error_);
    }
    // Append whatever the diagnostic engine said. On a pipeline failure this
    // is the only description there is; on a rejected flag it adds context to
    // error_'s summary.
    std::lock_guard<std::mutex> guard(diagnostics_mutex_);
    if (!diagnostics_.empty()) {
      if (!message.empty()) message.push_back('\n');
      message.append(diagnostics_);
    }
    return message;
  }

  std::mutex diagnostics_mutex_;
  std::string diagnostics_;

  void EnableCrashDumps(
      ArtifactDumper::Transaction* artifact_transaction) override {
    if (crash_dump_transaction_) return;
    crash_dump_transaction_ = artifact_transaction;
    ireeCompilerInvocationSetCrashHandler(
        inv_, /*genLocalReproducer=*/false,
        [](iree_compiler_output_t** outOutput,
           void* userData) -> iree_compiler_error_t* {
          auto* self = static_cast<IREECompilerJob*>(userData);
          auto maybePath = self->crash_dump_transaction_->AllocateArtifactPath(
              /*label=*/"crash_reproducer", /*extension=*/"mlir",
              /*index=*/self->crash_dump_count_++);
          if (!maybePath) {
            *outOutput = nullptr;
            return nullptr;
          }

          return ireeCompilerOutputOpenFile(maybePath->c_str(), outOutput);
        },
        this);
  }

  bool SetFlag(const char* flag) override {
    auto* error = ireeCompilerSessionSetFlags(session_, 1, &flag);
    if (error) {
      SetError(error);
      return false;
    }
    return true;
  }

  bool SetFlags(xla::CompileOptionsProto options) override {
    // Set extra options, overriding env variables if appropriate.
    for (auto [option, option_override] : options.env_option_overrides()) {
      std::string override_string;
      if (option_override.has_string_field()) {
        override_string = option_override.string_field();
      } else if (option_override.has_bool_field()) {
        override_string = option_override.bool_field() ? "true" : "false";
      } else if (option_override.has_int_field()) {
        override_string = std::to_string(option_override.int_field());
      } else if (option_override.has_double_field()) {
        override_string = std::to_string(option_override.double_field());
      } else {
        assert(false &&
               "option value should be of type string, bool, int, or double");
      }
      if (!SetFlag(("--" + option + "=" + override_string).c_str())) {
        return false;
      }
    }
    return true;
  }

  std::string GetFlags() override {
    std::string flags;
    ireeCompilerSessionGetFlags(
        session_, /*nonDefaultOnly=*/false,
        [](const char* flag, size_t length, void* userData) {
          std::string* capture_flags = static_cast<std::string*>(userData);
          if (!capture_flags->empty()) {
            capture_flags->append(" ");
          }
          capture_flags->append(flag, length);
        },
        &flags);
    return flags;
  }

  bool ParseSourceBuffer(const void* buffer, size_t length) override {
    iree_compiler_source_t* source;
    auto* error = ireeCompilerSourceWrapBuffer(
        session_, "<jit>", static_cast<const char*>(buffer), length,
        /*isNullTerminated=*/false, &source);
    if (error) {
      SetError(error);
      return false;
    }
    retained_sources_.push_back(source);

    return ireeCompilerInvocationParseSource(inv_, source);
  }

  std::unique_ptr<CompilerOutput> CompileStandardPipeline() override {
    if (!ireeCompilerInvocationPipeline(inv_, IREE_COMPILER_PIPELINE_STD)) {
      return nullptr;
    }

    iree_compiler_error_t* error = ireeCompilerOutputOpenMembuffer(&output_);
    if (error) {
      SetError(error);
      return nullptr;
    }

    // Output.
    error = ireeCompilerInvocationOutputVMBytecode(inv_, output_);
    if (error) {
      SetError(error);
      return nullptr;
    }

    // Map the data.
    void* output_data = nullptr;
    uint64_t size = -1;
    error = ireeCompilerOutputMapMemory(output_, &output_data, &size);
    if (error) {
      SetError(error);
      return nullptr;
    }

    // Transfer the output_ to MMapCompilerOutput since the mapping is only
    // valid for the life of the output.
    iree_compiler_output_t* local_output = output_;
    output_ = nullptr;
    return std::make_unique<MMapCompilerOutput>(local_output, output_data,
                                                size);
  }

 private:
  void SetError(iree_compiler_error_t* error) {
    if (error_) {
      ireeCompilerErrorDestroy(error_);
    }
    error_ = error;
  }

  iree_compiler_session_t* session_;
  iree_compiler_invocation_t* inv_;
  SessionRecycler session_recycler_;

  std::vector<iree_compiler_source_t*> retained_sources_;
  iree_compiler_error_t* error_ = nullptr;
  ArtifactDumper::Transaction* crash_dump_transaction_ = nullptr;
  int crash_dump_count_ = 0;

  // Output.
  iree_compiler_output_t* output_ = nullptr;
};

}  // namespace

std::unique_ptr<CompilerJob> IREECompiler::StartJob() {
  auto* session = ireeCompilerSessionCreate();
  auto* inv = ireeCompilerInvocationCreate(session);

  // Console diagnostics are kept so that warnings on an otherwise successful
  // compile still reach a human -- the f64-to-f32 demotion notice, for one,
  // which changes the public function signature and is worth seeing. Errors
  // are additionally captured below and returned to the caller, so they show
  // up twice: once on stderr and once attached to the failure. That is the
  // right trade for a plugin loaded inside another process, where stderr may
  // be interleaved with unrelated output or not shown at all.
  ireeCompilerInvocationEnableConsoleDiagnostics(inv);

  auto job = std::make_unique<IREECompilerJob>(
      session, inv, [](iree_compiler_session_t* session) {
        ireeCompilerSessionDestroy(session);
      });

  // Registered after the job exists, since the job is the callback's userData.
  // Safe for the invocation's whole life: the job owns the invocation and
  // destroys it in its own destructor.
  ireeCompilerInvocationEnableCallbackDiagnostics(
      inv, /*flags=*/0, &IREECompilerJob::OnDiagnostic, job.get());

  // The input here should be stablehlo if coming from JAX and xla if
  // importing from XLA HLO. Set to xla for now as it merely runs an
  // additional pass. We can flip to auto post more testing.
  if (!job->SetFlag("--iree-input-type=stablehlo_xla") ||
      !job->SetFlag("--iree-input-demote-i64-to-i32=false") ||
      !job->SetFlag("--iree-execution-model=async-external")) {
    error_message_ = job->GetErrorMessage();
    return nullptr;
  }

  // Propagate all options set via environment variable.
  for (auto arg : extra_options_) {
    if (!job->SetFlag(arg.c_str())) {
      error_message_ = job->GetErrorMessage();
      return nullptr;
    }
  }

  return job;
}

std::string IREECompiler::GetRevision() {
  std::string result;
  const char* revision = ireeCompilerGetRevision();
  result.append(revision[0] ? revision : "<unknown>");
  result.append(" (API version ");
  int packed_api_version = ireeCompilerGetAPIVersion();
  result.append(std::to_string(packed_api_version >> 16));
  result.append(".");
  result.append(std::to_string(packed_api_version & 0xffff));
  result.append(")");
  return result;
}

}  // namespace iree::pjrt
